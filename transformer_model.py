"""
transformer_model.py
--------------------
Lightweight Transformer encoder for GNSS spoofing detection.

Why Transformer?
  The dataset has ~8 channels (PRNs) measured simultaneously at each RX_time.
  During spoofing, the attacker broadcasts a single fake signal received by ALL
  channels — creating a characteristic cross-channel correlation pattern.

  Self-attention naturally captures this: the model can learn "if channel A and
  channel B both show anomalous Doppler at the same time, it's likely spoofing."

Architecture:
  Input  → (batch, seq_len, n_features)
  Linear projection → d_model
  Positional encoding (learnable)
  TransformerEncoder (4 layers, 4 heads, d_model=128, ff=256)
  GlobalMaxPool + GlobalAvgPool → concat → Dense → output
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Optional
import os

from lstm_model import SequenceDataset   # reuse dataset class


# ---------------------------------------------------------------------------
# Positional Encoding (learnable)
# ---------------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TransformerSpoofDetector(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Project raw features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = LearnablePositionalEncoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Dual pooling: max + mean → richer representation
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x)          # (batch, seq_len, d_model)

        max_pool, _ = x.max(dim=1)       # (batch, d_model)
        avg_pool = x.mean(dim=1)         # (batch, d_model)
        pooled = torch.cat([max_pool, avg_pool], dim=-1)  # (batch, d_model*2)

        return self.head(pooled).squeeze(-1)   # (batch,) logits


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TransformerTrainer:
    """
    Trains TransformerSpoofDetector.
    Same training loop as BiLSTMTrainer for consistency.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.2,
        lr: float = 5e-4,
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

        self.model = TransformerSpoofDetector(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=seq_len,
            dropout=dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "TransformerTrainer":

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.98)
        )
        # Warmup + cosine decay
        def lr_lambda(step):
            warmup = 200
            if step < warmup:
                return step / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (self.epochs * 100)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        train_ds = SequenceDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        best_f1 = -1
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
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
                scheduler.step()
                total_loss += loss.item() * len(batch_X)

            if X_val is not None:
                probs = self.predict_proba(X_val)
                preds = (probs >= 0.5).astype(int)
                val_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
                print(
                    f"  [Transformer] Epoch {epoch:02d}/{self.epochs}"
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
                        print(f"  [Transformer] Early stopping at epoch {epoch}")
                        break
            else:
                print(f"  [Transformer] Epoch {epoch:02d}  loss={total_loss/len(train_ds):.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [Transformer] Best val weighted F1: {best_f1:.4f}")

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
