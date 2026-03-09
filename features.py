"""
features.py
-----------
Physics-inspired + statistical feature engineering for GNSS anti-spoofing.

Feature groups:
  1. Correlator distortion   (EC/LC symmetry, S-curve shape)
  2. CN0 anomaly             (per-PRN z-score, absolute spike flags)
  3. Temporal dynamics       (rolling std, diff, 2nd-order diff per PRN)
  4. Timing mismatch         (RX_time vs TOW)
  5. Cross-channel Doppler   (deviation from channel consensus)
  6. Geometric consistency   (pseudorange spread across PRNs at same RX_time)
"""

import numpy as np
import pandas as pd
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [3, 5, 10, 20]
FEATURE_COLS = [
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
]


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns a copy of df with all engineered features appended.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Correlator Distortion
    # ------------------------------------------------------------------
    eps = 1e-9
    df["EC_LC_ratio"] = df["EC"] / (df["LC"] + eps)
    df["correlator_sym"] = np.abs(df["EC"] - df["LC"])
    df["EC_PC_ratio"] = df["EC"] / (df["PC"] + eps)
    df["LC_PC_ratio"] = df["LC"] / (df["PC"] + eps)
    # S-curve discriminator: (EC - LC) / (EC + LC)  → 0 for clean signals
    df["S_curve"] = (df["EC"] - df["LC"]) / (df["EC"] + df["LC"] + eps)
    # Peak ratio: (EC + LC) / (2*PC) should equal ~1 for genuine signal
    df["peak_ratio"] = (df["EC"] + df["LC"]) / (2 * df["PC"] + eps)
    # PIP/PQP interaction
    df["PIP_PQP_ratio"] = df["PIP"] / (df["PQP"] + eps)
    df["PIP_PQP_diff"] = df["PIP"] - df["PQP"]

    # ------------------------------------------------------------------
    # 2. CN0 Anomaly (per-PRN z-score)
    # ------------------------------------------------------------------
    prn_cn0_stats = (
        df.groupby("PRN")["CN0"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "CN0_prn_mean", "std": "CN0_prn_std"})
    )
    df = df.join(prn_cn0_stats, on="PRN")
    df["CN0_zscore"] = (df["CN0"] - df["CN0_prn_mean"]) / (df["CN0_prn_std"] + eps)
    df["CN0_high_flag"] = (df["CN0"] > 47).astype(int)   # empirical spoofing threshold
    df.drop(columns=["CN0_prn_mean", "CN0_prn_std"], inplace=True)

    # CN0 deviation from cross-PRN median at each RX_time
    cn0_time_med = df.groupby("RX_time")["CN0"].transform("median")
    df["CN0_time_deviation"] = df["CN0"] - cn0_time_med

    # ------------------------------------------------------------------
    # 3. Timing Mismatch
    # ------------------------------------------------------------------
    df["time_diff"] = df["RX_time"] - df["TOW_at_current_symbol_s"]

    # ------------------------------------------------------------------
    # 4. Cross-Channel Doppler Deviation
    # ------------------------------------------------------------------
    doppler_mean = df.groupby("RX_time")["Carrier_Doppler_hz"].transform("mean")
    df["doppler_deviation"] = np.abs(df["Carrier_Doppler_hz"] - doppler_mean)
    doppler_std = df.groupby("RX_time")["Carrier_Doppler_hz"].transform("std")
    df["doppler_channel_std"] = doppler_std.fillna(0)

    # ------------------------------------------------------------------
    # 5. Geometric Consistency (pseudorange spread per timestamp)
    # ------------------------------------------------------------------
    prange_std = df.groupby("RX_time")["Pseudorange_m"].transform("std")
    df["pseudorange_spread"] = prange_std.fillna(0)
    prange_mean = df.groupby("RX_time")["Pseudorange_m"].transform("mean")
    df["pseudorange_deviation"] = np.abs(df["Pseudorange_m"] - prange_mean)

    # ------------------------------------------------------------------
    # 6. Temporal Rolling Features (sort by PRN, then RX_time)
    # ------------------------------------------------------------------
    df = df.sort_values(["PRN", "RX_time"]).reset_index(drop=True)

    temporal_cols = [
        "Carrier_Doppler_hz",
        "Carrier_phase_cycles",
        "Pseudorange_m",
        "CN0",
        "TCD",
        "S_curve",
        "correlator_sym",
    ]

    for col in temporal_cols:
        grp = df.groupby("PRN")[col]

        # First-order difference (velocity of change)
        df[f"{col}_diff"] = grp.diff().fillna(0)

        # Second-order difference (acceleration — catches transition moments)
        df[f"{col}_diff2"] = df.groupby("PRN")[f"{col}_diff"].diff().fillna(0)

        for w in ROLLING_WINDOWS:
            df[f"{col}_roll_std_{w}"] = (
                grp.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
            )
            df[f"{col}_roll_mean_{w}"] = (
                grp.transform(lambda x: x.rolling(w, min_periods=1).mean()).fillna(0)
            )

    # ------------------------------------------------------------------
    # 7. Channel count per timestamp (fewer channels = suspicious)
    # ------------------------------------------------------------------
    ch_count = df.groupby("RX_time")["PRN"].transform("count")
    df["channel_count"] = ch_count

    # ------------------------------------------------------------------
    # 8. PRN-level label frequency (only for training aggregation)
    #    (not a leakage risk since it's PRN-level, not sample-level)
    # ------------------------------------------------------------------

    return df


def aggregate_to_timestamp(df: pd.DataFrame, label_col: str = None) -> pd.DataFrame:
    """
    Aggregate channel-level rows to one prediction per RX_time.

    Aggregation:
      - Numeric features: mean, std, max, min across channels
      - Label (if present): max across channels (1 if ANY channel spoofed)
    """
    # Columns that are metadata, not features
    meta_cols = {"PRN", "RX_time", "TOW_at_current_symbol_s"}
    if label_col:
        meta_cols.add(label_col)

    feature_cols = [c for c in df.columns if c not in meta_cols]

    agg_dict = {}
    for col in feature_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"]  = (col, "std")
        agg_dict[f"{col}_max"]  = (col, "max")
        agg_dict[f"{col}_min"]  = (col, "min")

    grouped = df.groupby("RX_time")
    result = grouped[feature_cols].agg(["mean", "std", "max", "min"])
    result.columns = ["_".join(c) for c in result.columns]
    result = result.reset_index()

    if label_col and label_col in df.columns:
        labels = df.groupby("RX_time")[label_col].max().reset_index()
        result = result.merge(labels, on="RX_time")

    return result


def get_sequence_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 20,
    label_col: str = None,
    stride: int = 1,
):
    """
    Build (X_seq, y_seq, timestamps) for LSTM/Transformer training.

    For each PRN track, create sliding windows of length `seq_len`.
    Label of window = label of the LAST element in the window.

    Returns:
      X: np.ndarray of shape (N, seq_len, n_features)
      y: np.ndarray of shape (N,)  — only if label_col provided
      ts: list of (PRN, RX_time) for the last element of each window
    """
    X_list, y_list, ts_list = [], [], []

    df = df.sort_values(["PRN", "RX_time"]).reset_index(drop=True)

    for prn, track in df.groupby("PRN"):
        track = track.reset_index(drop=True)
        vals = track[feature_cols].values.astype(np.float32)
        times = track["RX_time"].values

        for i in range(0, len(track) - seq_len + 1, stride):
            window = vals[i : i + seq_len]
            X_list.append(window)
            ts_list.append((prn, times[i + seq_len - 1]))
            if label_col and label_col in track.columns:
                y_list.append(int(track[label_col].iloc[i + seq_len - 1]))

    X = np.array(X_list, dtype=np.float32)
    ts = ts_list

    if y_list:
        y = np.array(y_list, dtype=np.int64)
        return X, y, ts
    return X, ts
