"""
qPCR anomaly detector v2: multi-channel conv autoencoder with local (per-cycle /
per-window) residual scoring.

How it works:
  * INPUT: 3 channels per well -- raw RFU, 1st derivative, 2nd derivative. The
    derivative channels exist on negative curves too (they are just ~0 noise).
    Each channel is asinh-scaled so a flat 0.02 RFU negative and a 4 RFU
    positive both stay resolvable in one model.
  * PHYSICS: first conv kernels are 7 then 5 cycles wide (spans the takeoff
    phase). Every conv uses REPLICATE padding and the decoder upsamples rather
    than using ConvTranspose, so cycles 1-5 (baseline) and 36-40 (plateau) are
    never mixed with fabricated zeros at the edges.
  * BOTTLENECK: a small latent vector (default 6-d) so the code stays small and
    the model cannot memorise curves.
  * SCORING: residuals are kept per channel, per cycle, and per sliding/phase
    window. Nothing is averaged over the whole curve. A single global statistic
    (worst cycle-or-window, each normalised by its own clean-data threshold)
    decides REVIEW, so the user never has to pick a window.
  * CLASS: negatives and positives have very different residual scales, so
    residual sd and thresholds are kept per class. The class comes from an
    injected class_fn (the RandomForest amp call in QPCRPipelineV2).

"""
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares

from sklearn.ensemble import RandomForestClassifier

from qpcr_pipeline import (CYCLE_COLS, N_CYCLES, CLINICAL_THRESHOLD, curve_features,
                           make_amp_labels)

CHANNELS = ("raw", "d1", "d2")
BASELINE = slice(0, 5)  # cycles 1-5, used to estimate per-channel noise scale

# Phase zones (1-indexed, inclusive) in addition to the sliding windows.
PHASE_ZONES = [("baseline 1-5", 1, 5), ("early 3-15", 3, 15), ("takeoff 10-25", 10, 25),
               ("mid 15-30", 15, 30), ("plateau 30-40", 30, 40)]
SLIDING_WIDTHS = (5, 8)


def make_windows(n_cycles=N_CYCLES, widths=SLIDING_WIDTHS, zones=PHASE_ZONES):
    """List of (name, start, end), 1-indexed inclusive. Multi-scale sliding
    windows at stride 1, then the fixed phase zones."""
    wins = []
    for w in widths:
        for s in range(1, n_cycles - w + 2):
            wins.append((f"w{w}: {s}-{s + w - 1}", s, s + w - 1))
    return wins + list(zones)


# ---------------------------------------------------------------- augmentation

def augment_clean(df, copies, seed=0):
    """Extra CLEAN curves for autoencoder training (never used for calibration).
      * negatives: cycles shuffled within the curve. Negative noise is i.i.d.
        across cycles (lag-1 autocorrelation ~ 0), so this is a new valid draw.
      * positives: shifted by 1-3 whole cycles (earlier/later Ct) with edge
        replication, and amplitude scaled 0.9-1.1x above baseline. Whole-cycle
        shifts keep the noise statistics intact (fractional shifts would smooth it).
    Returns (curves_df, is_pos)."""
    rng = np.random.default_rng(seed)
    X = df[CYCLE_COLS].to_numpy(dtype=np.float64)
    is_pos = df["true_label"].isin(("Positive", "Positive_Noisy")).to_numpy()
    out, out_pos = [], []
    for _ in range(copies):
        for x, p in zip(X, is_pos):
            if p:
                k = int(rng.choice([-3, -2, -1, 1, 2, 3]))
                idx = np.clip(np.arange(N_CYCLES) - k, 0, N_CYCLES - 1)
                y = x[idx]
                b = y[:5].mean()
                y = b + (y - b) * rng.uniform(0.9, 1.1)
            else:
                y = rng.permutation(x)
            out.append(y); out_pos.append(int(p))
    return pd.DataFrame(out, columns=CYCLE_COLS), np.array(out_pos)


# ---------------------------------------------------------------- channels

def build_channels(X):
    """(n, 40) raw RFU -> (n, 3, 40): raw, d1, d2. Derivatives come from a 3-pt
    smoothed curve padded by edge replication, so the smoother does not drag
    the first/last cycles toward zero."""
    X = np.asarray(X, dtype=np.float64)
    pad = np.pad(X, ((0, 0), (1, 1)), mode="edge")
    smoothed = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3
    d1 = np.gradient(smoothed, axis=1)
    d2 = np.gradient(d1, axis=1)
    return np.stack([X, d1, d2], axis=1)


class ChannelScaler:
    """asinh(channel / s_k), with s_k = 3 x the within-curve noise sd of channel k
    over the baseline cycles of the training curves. Linear near zero (noise
    stays visible), logarithmic for large values (positives don't swamp negatives)."""

    def fit(self, channels):
        base = channels[:, :, BASELINE]
        noise = (base - base.mean(axis=2, keepdims=True)).std(axis=(0, 2))
        self.s = 3.0 * noise + 1e-9
        return self

    def transform(self, channels):
        return np.arcsinh(channels / self.s[None, :, None])

    def inverse(self, scaled):
        return np.sinh(scaled) * self.s[None, :, None]


# ---------------------------------------------------------------- model

def _conv(c_in, c_out, k, stride, mode):
    return nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, padding_mode=mode)


class ConvAE3(nn.Module):
    """3-channel conv AE over the cycle axis. padding_mode='replicate' is the
    physics-respecting default; 'zeros' exists only to demonstrate the edge bias."""

    def __init__(self, latent_dim=6, padding_mode="replicate", n_cycles=N_CYCLES):
        super().__init__()
        m = padding_mode
        self.len_b = n_cycles // 4  # two stride-2 stages
        self.encoder = nn.Sequential(
            _conv(3, 16, 7, 1, m), nn.ReLU(),     # takeoff-scale receptive field
            _conv(16, 32, 5, 2, m), nn.ReLU(),
            _conv(32, 32, 5, 2, m), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.len_b, latent_dim),
        )
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, 32 * self.len_b), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            _conv(32, 32, 5, 1, m), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            _conv(32, 16, 5, 1, m), nn.ReLU(),
            _conv(16, 3, 7, 1, m),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(self.dec_fc(z).view(-1, 32, self.len_b))

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------- detector

class QPCRAnomalyV2:
    def __init__(self, epochs=150, lr=1e-3, weight_decay=1e-4, batch_size=32,
                 latent_dim=6, padding_mode="replicate",
                 unit_pct=99, flag_pct=95, random_state=42, class_fn=None):
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.latent_dim = batch_size, latent_dim
        self.padding_mode = padding_mode
        self.unit_pct, self.flag_pct = unit_pct, flag_pct
        self.random_state = random_state
        self.class_fn = class_fn  # df -> 0/1 per row; None = everything is class 0
        self.windows = make_windows()
        self.scaler = ChannelScaler()
        self.ae = None

    # ---- helpers
    def _prep(self, df):
        raw = df[CYCLE_COLS].to_numpy(dtype=np.float64)
        return self.scaler.transform(build_channels(raw))

    def _run(self, xs):
        """scaled (n,3,40) -> latent (n,d), recon (n,3,40)."""
        self.ae.eval()
        with torch.no_grad():
            x = torch.tensor(xs, dtype=torch.float32)
            z = self.ae.encode(x)
            r = self.ae.decode(z)
        return z.numpy(), r.numpy()

    def _win_err(self, cycle_err):
        return np.stack([cycle_err[:, a - 1:b].mean(axis=1) for _, a, b in self.windows], axis=1)

    def _cls(self, df):
        if self.class_fn is None:
            return np.zeros(len(df), dtype=int)
        return np.asarray(self.class_fn(df)).astype(int)

    # ---- fit
    def fit(self, train_df, verbose=True):
        """train_df: clean curves only; the AE never sees labels."""
        raw = train_df[CYCLE_COLS].to_numpy(dtype=np.float64)
        ch = build_channels(raw)
        self.scaler.fit(ch)
        xs = self.scaler.transform(ch).astype(np.float32)

        torch.manual_seed(self.random_state)
        self.ae = ConvAE3(self.latent_dim, self.padding_mode)
        opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X = torch.tensor(xs)
        self.losses = []
        for ep in range(self.epochs):
            self.ae.train()
            perm = torch.randperm(len(X))
            tot, nb = 0.0, 0
            for s in range(0, len(X), self.batch_size):
                b = X[perm[s:s + self.batch_size]]
                opt.zero_grad()
                loss = torch.mean((self.ae(b) - b) ** 2)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
            self.losses.append(tot / nb)
            if verbose and (ep + 1) % 25 == 0:
                print(f"  [{self.padding_mode}] epoch {ep + 1}/{self.epochs}  loss={self.losses[-1]:.5f}")

        return self

    # ---- calibrate on held-out CLEAN curves
    def calibrate(self, calib_df):
        """Sets, from clean held-out wells only (never test wells):
          - per-class, per-channel, per-cycle residual sd (for z-scoring)
          - per-class threshold for every cycle and every window (unit_pct)
          - global cutoff on the worst-unit ratio (flag_pct)
        Stats start pooled; a class with >= MIN_CLASS calibration wells then
        gets its own, so a sparse class falls back to pooled instead of noise."""
        MIN_CLASS = 20
        xs = self._prep(calib_df)
        z, rec = self._run(xs)
        cls = self._cls(calib_df)
        res = rec - xs

        pooled_sd = res.std(axis=0) + 1e-6
        self.res_sd = np.stack([pooled_sd, pooled_sd])
        for c in (0, 1):
            if (cls == c).sum() >= MIN_CLASS:
                self.res_sd[c] = res[cls == c].std(axis=0) + 1e-6

        cycle_err, win_err = self._errors(res, cls)
        pc_cyc = np.percentile(cycle_err, self.unit_pct, axis=0)
        pc_win = np.percentile(win_err, self.unit_pct, axis=0)
        self.thr_cycle = np.stack([pc_cyc, pc_cyc])
        self.thr_win = np.stack([pc_win, pc_win])
        self.class_counts = {int(c): int((cls == c).sum()) for c in (0, 1)}
        for c in (0, 1):
            m = cls == c
            if m.sum() >= MIN_CLASS:
                self.thr_cycle[c] = np.percentile(cycle_err[m], self.unit_pct, axis=0)
                self.thr_win[c] = np.percentile(win_err[m], self.unit_pct, axis=0)

        S = self._worst_ratio(cycle_err, win_err, cls)[0]
        self.flag_cut = float(np.percentile(S, self.flag_pct))
        return self

    def _errors(self, res, cls):
        z2 = (res / self.res_sd[cls]) ** 2          # (n,3,40)
        cycle_err = z2.mean(axis=1)                  # (n,40)
        return cycle_err, self._win_err(cycle_err)

    def _worst_ratio(self, cycle_err, win_err, cls):
        cr = cycle_err / self.thr_cycle[cls]
        wr = win_err / self.thr_win[cls]
        allr = np.concatenate([cr, wr], axis=1)
        return allr.max(axis=1), allr.argmax(axis=1), cr, wr

    def unit_span(self, u):
        """Unit index -> (first_cycle, last_cycle), 1-indexed inclusive."""
        if u < N_CYCLES:
            return u + 1, u + 1
        _, a, b = self.windows[u - N_CYCLES]
        return a, b

    # ---- score
    def score(self, df):
        """Everything is returned so it can be inspected: per-channel per-cycle
        z^2, per-cycle error, per-window error, their threshold ratios, the
        worst unit and its cycle span."""
        xs = self._prep(df)
        z, rec = self._run(xs)
        cls = self._cls(df)
        res = rec - xs
        z2 = (res / self.res_sd[cls]) ** 2
        cycle_err, win_err = self._errors(res, cls)
        S, arg, cr, wr = self._worst_ratio(cycle_err, win_err, cls)

        unit_names = [f"cycle {i}" for i in range(1, N_CYCLES + 1)] + [w[0] for w in self.windows]
        span = np.array([self.unit_span(int(u)) for u in arg])
        return dict(
            x_scaled=xs, recon_scaled=rec, latent=z, cls=cls,
            cycle_z2=z2, cycle_err=cycle_err, win_err=win_err,
            cycle_ratio=cr, win_ratio=wr, S=S, S_unit=arg, S_span=span,
            S_label=np.array([unit_names[i] for i in arg]),
            resid_flag=S > self.flag_cut,
            recon_raw=self.scaler.inverse(rec)[:, 0, :],
        )


# ---------------------------------------------------------------- materiality gate

def logistic(p, c):
    b, A, c0, k = p
    return b + A / (1 + np.exp(-(c - c0) / k))


def fit_curve(y):
    """Expected S-curve for one amplifying well: 4-parameter logistic fitted with a robust
    (soft-L1) loss so an artifact cannot drag the fit toward itself."""
    cyc = np.arange(1, N_CYCLES + 1)
    b0, top = np.median(y[:5]), np.median(y[-5:])
    A0 = max(top - b0, 0.2)
    above = np.where(y > b0 + A0 / 2)[0]
    c0 = cyc[above[0]] if len(above) else 20
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = least_squares(lambda p: logistic(p, cyc) - y, [b0, A0, c0, 1.5],
                              bounds=([-1, .05, 1, .3], [1, 10, 45, 6]),
                              loss="soft_l1", f_scale=0.05 + 0.02 * A0)
        return logistic(r.x, cyc)
    except Exception:  # noqa: BLE001 - a failed fit must not break a whole plate
        return np.full(N_CYCLES, np.median(y))


# ---------------------------------------------------------------- product wrapper

# Measured on held-out clean wells: false flags are all marginal (1.0-1.8x the calibrated
# 95th-percentile cut) while real artifacts sit around 11x. A 1.5x review limit cuts false
# alarms from ~8% / ~15% (neg / pos) to ~0.7% / ~2.6% at a cost of ~3 points of recall.
REVIEW_MULT = 1.5   # a well is REVIEW above REVIEW_MULT x the calibrated flag cut
MATERIAL_FRAC = 0.05   # AMP wells: min departure from the fitted curve, as a share of curve height
HIGH_MULT = 2.0     # HIGH priority above HIGH_MULT x the review limit (= 3x the calibrated cut)


class QPCRPipelineV2:
    """What the dashboard talks to: RandomForest amp call (clinical question) +
    class-aware v2 autoencoder (quality question). Same results columns as the
    old QPCRPipeline, plus review tier and the anomalous cycle span."""

    def __init__(self, epochs=100, batch_size=32, calib_frac=0.15, random_state=42, augment=0, **ae_kw):
        self.epochs, self.batch_size, self.augment = epochs, batch_size, augment
        self.calib_frac, self.random_state, self.ae_kw = calib_frac, random_state, ae_kw
        self.clf, self.ae = None, None

    def _amp_pred(self, df):
        return self.clf.predict(curve_features(df).to_numpy())

    def fit(self, train_df, verbose=True):
        """train_df: clean curves with true_label (Positive/Negative_Clean)."""
        y = make_amp_labels(train_df)
        rng = np.random.default_rng(self.random_state)
        calib_idx = np.concatenate([
            rng.choice(np.where(y == c)[0], int(self.calib_frac * (y == c).sum()), replace=False)
            for c in (0, 1)])
        mask = np.zeros(len(train_df), bool)
        mask[calib_idx] = True
        fit_df, calib_df = train_df[~mask].reset_index(drop=True), train_df[mask].reset_index(drop=True)

        if verbose:
            print(f"Training classifier on {len(fit_df)} wells...")
        self.feature_names = list(curve_features(fit_df).columns)
        self.clf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                          random_state=self.random_state)
        self.clf.fit(curve_features(fit_df).to_numpy(), make_amp_labels(fit_df))

        if verbose:
            print("Training autoencoder...")
        self.ae = QPCRAnomalyV2(epochs=self.epochs, batch_size=self.batch_size,
                                random_state=self.random_state, class_fn=self._amp_pred, **self.ae_kw)
        ae_df = fit_df
        if self.augment:
            aug_df, _ = augment_clean(fit_df, self.augment, self.random_state)
            ae_df = pd.concat([fit_df[CYCLE_COLS], aug_df], ignore_index=True)
        self.n_train = len(fit_df)
        self.n_ae_train = len(ae_df)
        self.ae.fit(ae_df, verbose=verbose).calibrate(calib_df)
        return self

    def predict(self, df):
        """Returns (results, reconstructions, extras). extras carries per-cycle
        arrays the curve viewer needs (hot cycles, ratios)."""
        if self.ae is None or self.clf is None:
            raise RuntimeError("Call fit() before predict().")
        sc = self.ae.score(df)
        X = curve_features(df).to_numpy()
        amp_pred = self.clf.predict(X)
        proba = self.clf.predict_proba(X)[:, 1]
        crossed = (df[CYCLE_COLS].to_numpy(dtype=np.float64) > CLINICAL_THRESHOLD).any(axis=1)

        review_cut = self.ae.flag_cut * REVIEW_MULT
        flagged = sc["S"] > review_cut
        # materiality gate: an amplifying well is only worth a review if it departs from its own
        # fitted S-curve by a visible share of the curve height (small offsets/bumps don't change the result)
        Y = df[CYCLE_COLS].to_numpy(dtype=np.float64)
        for i in np.where(flagged & (amp_pred == 1))[0]:
            height = max(np.median(Y[i, -5:]) - np.median(Y[i, :5]), 0.2)
            if np.abs(Y[i] - fit_curve(Y[i])).max() / height < MATERIAL_FRAC:
                flagged[i] = False
        ratio = np.where(flagged, sc["S"] / review_cut, np.minimum(sc["S"] / review_cut, 1.0))
        tier = np.where(ratio > HIGH_MULT, "HIGH", np.where(ratio > 1, "REVIEW", "OK"))
        borderline = (pd.Series(proba).between(0.35, 0.65).to_numpy()) & (tier == "OK")
        tier = np.where(borderline, "BORDERLINE", tier)

        results = pd.DataFrame({
            "call": np.where(amp_pred == 1, "AMP", "NO AMP"),
            "confidence": proba,
            "anomaly": flagged,
            "anomaly_score": sc["S"],
            "crossed_threshold": crossed,
            "review_flag": tier,
            "worst_region": [f"{a}" if a == b else f"{a}–{b}" for a, b in sc["S_span"]],
            "span_start": sc["S_span"][:, 0], "span_end": sc["S_span"][:, 1],
        }, index=df.index)
        for col in ("well", "row", "col", "true_label", "true_ct", "has_anomaly", "false_positive_risk"):
            if col in df.columns:
                results.insert(0, col, df[col].values)

        extras = dict(cycle_ratio=sc["cycle_ratio"], hot=sc["cycle_ratio"] > 1.0,
                      flag_cut=review_cut)
        return results, sc["recon_raw"], extras
