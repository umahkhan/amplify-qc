"""
Train the production qPCR model once and save it to qpcr_model.pkl, so the
dashboard only ever runs inference.

    python train_model.py [--train train_normal.csv] [--out qpcr_model.pkl]

The model bundles the RandomForest amp call, the 3-channel autoencoder, the
per-class thresholds, and the training metadata shown in the app. Re-run this
whenever the training data or qpcr_v2.py changes.
"""
import argparse
import datetime as dt

import pandas as pd
import torch

from qpcr_v2 import QPCRPipelineV2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train_normal.csv")
    p.add_argument("--out", default="qpcr_model.pkl")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--augment", type=int, default=2, help="extra synthetic copies per real training curve")
    p.add_argument("--calib-frac", type=float, default=0.25, help="share of real clean wells held out for thresholds")
    args = p.parse_args()

    df = pd.read_csv(args.train)
    pipe = QPCRPipelineV2(epochs=args.epochs, batch_size=32, calib_frac=args.calib_frac,
                          augment=args.augment).fit(df, verbose=True)
    pipe.meta = dict(
        trained_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_file=args.train, n_real_wells=len(df), n_fit_wells=pipe.n_train,
        n_ae_curves=pipe.n_ae_train, epochs=args.epochs, augment=args.augment,
        flag_cut=pipe.ae.flag_cut, calib_counts=pipe.ae.class_counts,
        final_loss=float(pipe.ae.losses[-1]),
    )
    torch.save(pipe, args.out)
    print(f"\nSaved {args.out}: {pipe.meta}")


if __name__ == "__main__":
    main()
