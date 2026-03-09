"""
xgb_model.py
------------
XGBoost classifier for GNSS spoofing detection on aggregated tabular features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os


class XGBDetector:
    """
    XGBoost-based GNSS spoofing detector with OOF prediction support.
    """

    def __init__(
        self,
        n_estimators: int = 700,
        max_depth: int = 7,
        learning_rate: float = 0.04,
        subsample: float = 0.8,
        colsample_bytree: float = 0.75,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.5,
        n_splits: int = 5,
        random_state: int = 42,
        scale_pos_weight: float = None,   # set automatically if None
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_state,
        )
        self.n_splits = n_splits
        self.random_state = random_state
        self._scale_pos_weight = scale_pos_weight
        self.models = []
        self.feature_names_ = None

    def _make_model(self, spw: float) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(scale_pos_weight=spw, **self.params)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit with stratified K-fold cross-validation.
        Returns OOF probability predictions (useful for stacking).
        """
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float32)

        spw = self._scale_pos_weight
        if spw is None:
            neg = (y == 0).sum()
            pos = (y == 1).sum()
            spw = neg / max(pos, 1)
        self.spw_ = spw

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        oof_probs = np.zeros(len(y), dtype=np.float32)
        fold_f1s = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y)):
            X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = self._make_model(spw)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=50,
            )
            self.models.append(model)

            probs = model.predict_proba(X_va)[:, 1]
            oof_probs[va_idx] = probs
            preds = (probs >= 0.5).astype(int)
            f1 = f1_score(y_va, preds, average="weighted")
            fold_f1s.append(f1)
            print(f"  [XGB] Fold {fold+1}/{self.n_splits} — Weighted F1: {f1:.4f}")

        print(f"  [XGB] Mean CV Weighted F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        self.oof_probs_ = oof_probs
        return oof_probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average probabilities across all fold models."""
        X_arr = X[self.feature_names_].values.astype(np.float32)
        probs = np.mean(
            [m.predict_proba(X_arr)[:, 1] for m in self.models], axis=0
        )
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBDetector":
        return joblib.load(path)

    def feature_importance_df(self) -> pd.DataFrame:
        """Average feature importance across fold models."""
        importances = np.mean(
            [m.feature_importances_ for m in self.models], axis=0
        )
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
