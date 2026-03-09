"""
train_predict.py
----------------
Main entry point for the GNSS Anti-Spoofing Hybrid Ensemble pipeline.

Usage:
    python src/train_predict.py \
        --train   data/train.csv \
        --test    data/test.csv \
        --submission data/submission_format.csv \
        --output  outputs/submission.csv \
        --mode    [full | xgb_only | seq_only]
        --device  [cpu | cuda]

Modes:
  full      — Full hybrid ensemble (XGBoost + BiLSTM + Transformer) [default]
  xgb_only  — XGBoost only (fast, good baseline)
  seq_only  — BiLSTM + Transformer only (best for temporal patterns)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))

from features import build_features, aggregate_to_timestamp, get_sequence_data
from xgb_model import XGBDetector
from ensemble import HybridEnsemble


def run_xgb_only(args):
    """Fast XGBoost-only pipeline."""
    print("\n=== XGBoost-Only Mode ===")

    print("[1/4] Loading and engineering features...")
    df_train = pd.read_csv(args.train)
    df_train_feat = build_features(df_train, is_train=True)
    df_agg = aggregate_to_timestamp(df_train_feat, label_col=args.label)

    label_col = args.label
    X_train = df_agg.drop(columns=["RX_time", label_col], errors="ignore")
    y_train = df_agg[label_col].values
    timestamps_train = df_agg["RX_time"].values

    print("[2/4] Training XGBoost with cross-validation...")
    detector = XGBDetector()
    oof_probs = detector.fit(X_train, y_train)

    print("[3/4] OOF evaluation:")
    oof_preds = (oof_probs >= 0.5).astype(int)
    print(classification_report(y_train, oof_preds, target_names=["Genuine", "Spoofed"]))

    print("[4/4] Generating test predictions...")
    df_test = pd.read_csv(args.test)
    df_test_feat = build_features(df_test, is_train=False)
    df_test_agg = aggregate_to_timestamp(df_test_feat)
    X_test = df_test_agg.drop(columns=["RX_time"], errors="ignore")
    timestamps_test = df_test_agg["RX_time"].values

    probs_test = detector.predict_proba(X_test)
    preds_test = (probs_test >= 0.5).astype(int)

    sub_fmt = pd.read_csv(args.submission)
    submission = pd.DataFrame({
        sub_fmt.columns[0]: timestamps_test,
        sub_fmt.columns[1]: preds_test,
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"\n✓ Submission saved: {args.output}")
    print(f"  Predicted spoofed: {preds_test.sum()} / {len(preds_test)} timestamps ({100*preds_test.mean():.1f}%)")

    print("\nTop 20 Feature Importances:")
    print(detector.feature_importance_df().head(20).to_string(index=False))


def run_full_ensemble(args):
    """Full hybrid ensemble pipeline."""
    print("\n=== Full Hybrid Ensemble Mode ===")
    print("  Models: XGBoost + BiLSTM + Transformer → Logistic Regression meta-learner")

    ensemble = HybridEnsemble(
        label_col=args.label,
        device=args.device,
    )
    ensemble.fit(args.train)
    ensemble.predict(args.test, args.submission, args.output)
    ensemble.save(os.path.join(os.path.dirname(args.output), "saved_model"))


def main():
    parser = argparse.ArgumentParser(description="GNSS Anti-Spoofing Detection Pipeline")
    parser.add_argument("--train",       required=True, help="Path to train CSV")
    parser.add_argument("--test",        required=True, help="Path to test CSV")
    parser.add_argument("--submission",  required=True, help="Path to submission format CSV")
    parser.add_argument("--output",      default="outputs/submission.csv", help="Output CSV path")
    parser.add_argument("--label",       default="Label", help="Label column name")
    parser.add_argument(
        "--mode",
        choices=["full", "xgb_only", "seq_only"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for PyTorch models (auto-detected if not specified)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  GNSS Anti-Spoofing Detection — Hybrid Ensemble Pipeline")
    print(f"{'='*60}")
    print(f"  Train:      {args.train}")
    print(f"  Test:       {args.test}")
    print(f"  Output:     {args.output}")
    print(f"  Mode:       {args.mode}")
    print(f"  Label col:  {args.label}")
    print(f"{'='*60}")

    if args.mode == "xgb_only":
        run_xgb_only(args)
    else:
        run_full_ensemble(args)


if __name__ == "__main__":
    main()
