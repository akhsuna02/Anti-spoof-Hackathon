"""
lstm_model.py
-------------
Bidirectional LSTM for GNSS spoofing detection using per-PRN temporal sequences.

Architecture:
  Input  → (batch, seq_len, n_features)
  BiLSTM (128 units) → LayerNorm → Dropout
  BiLSTM (64  units) → LayerNorm → Dropout
  GlobalMaxPool over time
  Dense(64, relu) → Dropout
  Dense(1, sigmoid)

Why BiLSTM?
  - Spoofing is a sustained temporal block; LSTM captures sequential dependencies
  - Bidirectional: looks at both past AND future context for each timestep
  - Captures transition moments (start/end of attack) better than tree models
  - GlobalMaxPool: "Did spoofing occur at ANY point in this window?" — very effective
    given that the attack is a contiguous block
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from typing import Optional, Tuple
import os


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BiLSTMSpoofDetector(nn.Module):
    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1 = nn.LayerNorm(hidden1 * 2)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden1 * 2,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln2 = nn.LayerNorm(hidden2 * 2)
        self.drop2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.GELU()
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)          # (batch, seq_len, hidden1*2)
        out = self.drop1(self.ln1(out))

        out, _ = self.lstm2(out)        # (batch, seq_len, hidden2*2)
        out = self.drop2(self.ln2(out))

        # Global max pooling across time — captures worst-case anomaly
        out, _ = out.max(dim=1)         # (batch, hidden2*2)

        out = self.drop3(self.act(self.fc1(out)))
        out = self.fc2(out).squeeze(-1)  # (batch,)
        return out                        # raw logits


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class BiLSTMTrainer:
    """
    Trains BiLSTMSpoofDetector with:
      - Weighted BCE loss (class imbalance)
      - Cosine LR annealing
      - Early stopping on validation weighted F1
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 512,
        patience: int = 7,
        device: str = None,
    ):
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMSpoofDetector(n_features, hidden1, hidden2, dropout).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "BiLSTMTrainer":

        # Compute class weight
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_ds = SequenceDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        best_f1 = -1
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            self.model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_dl:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_X)
            scheduler.step()

            # ---- Validate ----
            if X_val is not None:
                probs = self.predict_proba(X_val)
                preds = (probs >= 0.5).astype(int)
                val_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)

                print(
                    f"  [LSTM] Epoch {epoch:02d}/{self.epochs}"
                    f"  loss={total_loss/len(train_ds):.4f}"
                    f"  val_weighted_F1={val_f1:.4f}"
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  [LSTM] Early stopping at epoch {epoch}")
                        break
            else:
                print(f"  [LSTM] Epoch {epoch:02d}  loss={total_loss/len(train_ds):.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [LSTM] Best val weighted F1: {best_f1:.4f}")

        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        ds = SequenceDataset(X)
        dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
        all_probs = []
        for batch_X in dl:
            batch_X = batch_X.to(self.device)
            logits = self.model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self
