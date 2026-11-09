"""
Shared qPCR helpers: curve constants, the AMP / NO AMP label rule, the shape features the
RandomForest amplification call is built on, and the ground-truth benchmark shown in the
app's validation panel when a plate carries true labels.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

N_CYCLES = 40
CYCLE_COLS = [f"c{i}" for i in range(1, N_CYCLES + 1)]
CLINICAL_THRESHOLD = 0.15
AMP_LABELS = ("Positive", "Positive_Noisy")


# ---------------------------------------------------------------- data utils

def make_amp_labels(df):
    """Binary target: 1 if the well truly amplified, 0 otherwise."""
    return df["true_label"].isin(AMP_LABELS).astype(int).to_numpy()


def curve_features(df, cycle_cols=CYCLE_COLS):
    """Summarize each curve into interpretable scalars for the tree model.
    Trees don't handle raw time-series well -- they have no notion that cycle
    15 is adjacent to cycle 16 -- so we hand them shape statistics instead."""
    X = df[cycle_cols].to_numpy(dtype=np.float64)
    smoothed = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, X)
    d1 = np.gradient(smoothed, axis=1)

    return pd.DataFrame({
        "baseline_mean": X[:, :5].mean(axis=1),
        "baseline_std": X[:, :5].std(axis=1),
        "plateau": X[:, -5:].mean(axis=1),
        "max_val": X.max(axis=1),
        "min_val": X.min(axis=1),
        "range": X.max(axis=1) - X.min(axis=1),
        "max_cycle": X.argmax(axis=1) + 1,
        "max_slope": d1.max(axis=1),
        "max_slope_cycle": d1.argmax(axis=1) + 1,
        "end_minus_start": X[:, -5:].mean(axis=1) - X[:, :5].mean(axis=1),
        "residual_noise": (X - smoothed).std(axis=1),
        "n_above_thresh": (X > CLINICAL_THRESHOLD).sum(axis=1),
    }, index=df.index)


# ---------------------------------------------------------------- benchmarks

def sens_spec(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fn + 1e-9), tn / (tn + fp + 1e-9)


def benchmark(results, test_df, verbose=True):
    """Compares the ML classifier against a naive threshold-crossing caller,
    and reports anomaly-detection precision/recall. Requires ground-truth
    columns; returns None if they're absent (i.e. real unlabeled data)."""
    if "true_label" not in test_df.columns:
        if verbose:
            print("No ground-truth labels in this file -- skipping benchmark.")
        return None

    y_true = make_amp_labels(test_df)
    y_ml = (results["call"] == "AMP").astype(int).to_numpy()
    y_naive = results["crossed_threshold"].astype(int).to_numpy()

    sens_ml, spec_ml = sens_spec(y_true, y_ml)
    sens_nv, spec_nv = sens_spec(y_true, y_naive)

    out = {
        "sens_ml": sens_ml, "spec_ml": spec_ml,
        "sens_naive": sens_nv, "spec_naive": spec_nv,
        "d_sens": (sens_ml - sens_nv) * 100, "d_spec": (spec_ml - spec_nv) * 100,
    }

    if "has_anomaly" in test_df.columns:
        truth = test_df["has_anomaly"].to_numpy(dtype=bool)
        pred = results["anomaly"].to_numpy(dtype=bool)
        tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
        out["anomaly_precision"] = tp / (tp + fp + 1e-9)
        out["anomaly_recall"] = tp / (tp + fn + 1e-9)
        out["anomaly_tp"], out["anomaly_fp"], out["anomaly_fn"] = int(tp), int(fp), int(fn)

    if verbose:
        print("\n--- Clinical call: ML classifier vs naive threshold caller ---")
        print(f"  ML classifier    sensitivity={sens_ml:.3f}  specificity={spec_ml:.3f}")
        print(f"  Naive threshold  sensitivity={sens_nv:.3f}  specificity={spec_nv:.3f}")
        print(f"  Delta            sensitivity={out['d_sens']:+.1f} pts  specificity={out['d_spec']:+.1f} pts")
        print("\n" + classification_report(y_true, y_ml, target_names=["Non-amp", "Amp"]))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, y_ml))
        if "anomaly_precision" in out:
            print("\n--- Anomaly detection ---")
            print(f"  precision={out['anomaly_precision']:.3f}  recall={out['anomaly_recall']:.3f}  "
                  f"TP={out['anomaly_tp']} FP={out['anomaly_fp']} FN={out['anomaly_fn']}")
        if "false_positive_risk" in test_df.columns:
            fooled = test_df["false_positive_risk"].to_numpy(dtype=bool)
            saved = (fooled & (y_ml == 0)).sum()
            print(f"\n  Wells where noise crossed the threshold (naive caller calls these AMP): {fooled.sum()}")
            print(f"  ML classifier correctly called NO AMP: {saved} / {fooled.sum()}")

    return out
