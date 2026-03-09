"""
ensemble.py
-----------
Hybrid stacking ensemble combining:
  1. XGBoost on aggregated tabular features
  2. BiLSTM on per-PRN temporal sequences
  3. Transformer on per-PRN temporal sequences
  → Meta-learner: Logistic Regression on OOF predictions

Why stacking?
  - XGBoost excels at non-linear feature interactions (tabular data)
  - BiLSTM captures sequential dependencies and temporal transitions
  - Transformer captures cross-timestep attention patterns
  - Each model sees the problem from a different inductive bias
  - A meta-learner learns the OPTIMAL combination

Stacking correctly avoids data leakage by using Out-of-Fold (OOF) predictions
for training the meta-learner, and only applies test predictions from models
trained on the full training set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from features import build_features, aggregate_to_timestamp, get_sequence_data
from xgb_model import XGBDetector
from lstm_model import BiLSTMTrainer
from transformer_model import TransformerTrainer


# ---------------------------------------------------------------------------
# Sequence feature columns (used by LSTM and Transformer)
# ---------------------------------------------------------------------------
SEQ_FEATURE_COLS = [
    "Carrier_Doppler_hz",
    "Pseudorange_m",
    "Carrier_phase_cycles",
    "EC",
    "LC",
    "PC",
    "PIP",
    "PQP",
    "TCD",
    "CN0",
    # Engineered
    "S_curve",
    "correlator_sym",
    "EC_LC_ratio",
    "peak_ratio",
    "CN0_zscore",
    "time_diff",
    "doppler_deviation",
    "Carrier_Doppler_hz_diff",
    "Carrier_phase_cycles_diff",
    "CN0_diff",
    "S_curve_diff",
]

SEQ_LEN = 20  # sliding window length (timesteps)


class HybridEnsemble:
    """
    Full hybrid stacking pipeline for GNSS spoofing detection.
    """

    def __init__(
        self,
        label_col: str = "Label",
        seq_len: int = SEQ_LEN,
        n_splits: int = 5,
        random_state: int = 42,
        device: str = None,
    ):
        self.label_col = label_col
        self.seq_len = seq_len
        self.n_splits = n_splits
        self.random_state = random_state
        self.device = device

        self.xgb = XGBDetector(n_splits=n_splits, random_state=random_state)
        self.meta_scaler = StandardScaler()
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _filter_seq_features(self, df: pd.DataFrame) -> list:
        return [c for c in SEQ_FEATURE_COLS if c in df.columns]

    def _seq_proba_to_timestamp(
        self,
        seq_probs: np.ndarray,
        seq_ts: list,
        all_timestamps: np.ndarray,
        agg: str = "max",
    ) -> np.ndarray:
        """
        Map window-level probabilities back to timestamp-level.
        For each timestamp, aggregate all windows whose last element == that timestamp.
        """
        ts_to_probs = {t: [] for t in all_timestamps}
        for prob, (prn, ts) in zip(seq_probs, seq_ts):
            if ts in ts_to_probs:
                ts_to_probs[ts].append(prob)

        result = np.zeros(len(all_timestamps), dtype=np.float32)
        for i, ts in enumerate(all_timestamps):
            vals = ts_to_probs[ts]
            if vals:
                result[i] = max(vals) if agg == "max" else np.mean(vals)
        return result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train_path: str):
        """
        Full training pipeline:
          1. Feature engineering
          2. Timestamp aggregation for XGBoost
          3. Sequence extraction for LSTM + Transformer
          4. OOF predictions from all three base models
          5. Train meta-learner
        """
        print("\n[Ensemble] Loading and engineering features...")
        df_raw = pd.read_csv(train_path)
        df = build_features(df_raw, is_train=True)

        # ---- XGBoost (timestamp-level aggregated features) ----
        print("\n[Ensemble] Preparing tabular (XGBoost) features...")
        df_agg = aggregate_to_timestamp(df, label_col=self.label_col)
        drop_cols = ["RX_time", self.label_col]
        X_tab = df_agg.drop(columns=[c for c in drop_cols if c in df_agg.columns])
        y_ts = df_agg[self.label_col].values
        timestamps = df_agg["RX_time"].values

        print("\n[Ensemble] Training XGBoost...")
        xgb_oof = self.xgb.fit(X_tab, y_ts)   # OOF probs at timestamp level

        # ---- Sequence data for LSTM & Transformer ----
        print("\n[Ensemble] Extracting sequence data...")
        seq_feat_cols = self._filter_seq_features(df)
        X_seq, y_seq, seq_ts = get_sequence_data(
            df, seq_feat_cols, seq_len=self.seq_len, label_col=self.label_col, stride=1
        )
        n_features = len(seq_feat_cols)
        print(f"  Sequences: {X_seq.shape}, label distribution: {y_seq.mean():.3f}")

        # Normalize sequences
        X_flat = X_seq.reshape(-1, n_features)
        self.seq_scaler = StandardScaler().fit(X_flat)
        X_seq_norm = self.seq_scaler.transform(X_flat).reshape(X_seq.shape)

        # ---- OOF for LSTM ----
        print("\n[Ensemble] Training BiLSTM (OOF)...")
        lstm_oof_seq = self._oof_seq_model("lstm", X_seq_norm, y_seq, n_features)
        lstm_oof_ts = self._seq_proba_to_timestamp(lstm_oof_seq, seq_ts, timestamps)

        # ---- OOF for Transformer ----
        print("\n[Ensemble] Training Transformer (OOF)...")
        tfm_oof_seq = self._oof_seq_model("transformer", X_seq_norm, y_seq, n_features)
        tfm_oof_ts = self._seq_proba_to_timestamp(tfm_oof_seq, seq_ts, timestamps)

        # ---- Train full models on ALL data ----
        print("\n[Ensemble] Training FULL BiLSTM on all data...")
        split = int(0.85 * len(X_seq_norm))
        self.lstm_trainer = BiLSTMTrainer(
            n_features=n_features, seq_len=self.seq_len, device=self.device
        )
        self.lstm_trainer.fit(
            X_seq_norm[:split], y_seq[:split],
            X_seq_norm[split:], y_seq[split:]
        )

        print("\n[Ensemble] Training FULL Transformer on all data...")
        self.tfm_trainer = TransformerTrainer(
            n_features=n_features, seq_len=self.seq_len, device=self.device
        )
        self.tfm_trainer.fit(
            X_seq_norm[:split], y_seq[:split],
            X_seq_norm[split:], y_seq[split:]
        )

        # ---- Meta-learner ----
        print("\n[Ensemble] Training meta-learner (Logistic Regression)...")
        meta_X = np.column_stack([xgb_oof, lstm_oof_ts, tfm_oof_ts])
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)
        self.meta_model.fit(meta_X_scaled, y_ts)

        meta_preds = self.meta_model.predict(meta_X_scaled)
        final_f1 = f1_score(y_ts, meta_preds, average="weighted")
        print(f"\n[Ensemble] ✓ Meta-learner OOF Weighted F1: {final_f1:.4f}")

        # Store for inference
        self._seq_feat_cols = seq_feat_cols
        self._n_features = n_features
        return self

    def _oof_seq_model(
        self, model_type: str, X_seq: np.ndarray, y_seq: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Generate OOF predictions from sequence model using stratified KFold."""
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        oof_probs = np.zeros(len(y_seq), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_seq, y_seq)):
            if model_type == "lstm":
                trainer = BiLSTMTrainer(
                    n_features=n_features, seq_len=self.seq_len, device=self.device
                )
            else:
                trainer = TransformerTrainer(
                    n_features=n_features, seq_len=self.seq_len, device=self.device
                )

            trainer.fit(
                X_seq[tr_idx], y_seq[tr_idx],
                X_seq[va_idx], y_seq[va_idx]
            )
            oof_probs[va_idx] = trainer.predict_proba(X_seq[va_idx])
            print(f"  [{model_type.upper()}] OOF fold {fold+1}/{self.n_splits} done")

        return oof_probs

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        test_path: str,
        submission_format_path: str,
        output_path: str,
    ):
        """
        Generate predictions for the test set and write submission CSV.
        """
        print("\n[Ensemble] Loading test data...")
        df_test_raw = pd.read_csv(test_path)
        df_test = build_features(df_test_raw, is_train=False)

        # ---- XGBoost predictions ----
        print("[Ensemble] XGBoost inference...")
        df_test_agg = aggregate_to_timestamp(df_test)
        X_tab_test = df_test_agg.drop(
            columns=[c for c in ["RX_time"] if c in df_test_agg.columns]
        )
        timestamps_test = df_test_agg["RX_time"].values
        xgb_probs = self.xgb.predict_proba(X_tab_test)

        # ---- Sequence predictions ----
        print("[Ensemble] Extracting test sequences...")
        X_seq_test, seq_ts_test = get_sequence_data(
            df_test, self._seq_feat_cols, seq_len=self.seq_len, stride=1
        )
        X_flat = X_seq_test.reshape(-1, self._n_features)
        X_seq_test_norm = self.seq_scaler.transform(X_flat).reshape(X_seq_test.shape)

        print("[Ensemble] BiLSTM inference...")
        lstm_probs_seq = self.lstm_trainer.predict_proba(X_seq_test_norm)
        lstm_probs_ts = self._seq_proba_to_timestamp(
            lstm_probs_seq, seq_ts_test, timestamps_test
        )

        print("[Ensemble] Transformer inference...")
        tfm_probs_seq = self.tfm_trainer.predict_proba(X_seq_test_norm)
        tfm_probs_ts = self._seq_proba_to_timestamp(
            tfm_probs_seq, seq_ts_test, timestamps_test
        )

        # ---- Meta-learner ----
        print("[Ensemble] Meta-learner prediction...")
        meta_X = np.column_stack([xgb_probs, lstm_probs_ts, tfm_probs_ts])
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        final_preds = self.meta_model.predict(meta_X_scaled)

        # ---- Write submission ----
        sub_fmt = pd.read_csv(submission_format_path)
        submission = pd.DataFrame({
            sub_fmt.columns[0]: timestamps_test,
            sub_fmt.columns[1]: final_preds,
        })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"\n[Ensemble] ✓ Submission saved to: {output_path}")
        print(f"  Predicted spoofed: {final_preds.sum()} / {len(final_preds)} timestamps")
        return submission

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self, os.path.join(dir_path, "ensemble.pkl"))
        print(f"[Ensemble] Saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: str) -> "HybridEnsemble":
        return joblib.load(os.path.join(dir_path, "ensemble.pkl"))
