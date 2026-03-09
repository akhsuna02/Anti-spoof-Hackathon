# GNSS Anti-Spoofing Detection — Hybrid Ensemble
### NyneOS Hackathon · Kaizen 2026 · IIT Delhi

---

## Problem Understanding

**GNSS (GPS/GLONASS/Galileo)** signals are the backbone of aviation, maritime navigation, autonomous drones, financial timestamps, and national infrastructure.

These signals travel from satellites ~20,000 km away and arrive at receivers as extremely weak signals broadcast over open radio channels. A **spoofer** can transmit a stronger counterfeit signal that overrides the real one, causing the receiver to compute a wrong:

- **Position** — misleading a drone, ship, or aircraft to the wrong location
- **Time** — corrupting financial timestamps and telecom synchronization
- **Velocity** — triggering incorrect trajectory corrections in autonomous systems

**Our goal:** An AI system that analyzes incoming GNSS channel data in real-time and raises an alarm when spoofing is detected — without generating any spoofing attacks itself.

---

## Setup Instructions

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/gnss-antispoofing.git
cd gnss-antispoofing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset files
mkdir -p data outputs
# Put train.csv, test.csv, submission_format.csv in data/

# 4a. Run full hybrid ensemble (best accuracy)
python src/train_predict.py \
    --train data/train.csv \
    --test  data/test.csv \
    --submission data/submission_format.csv \
    --output outputs/submission.csv \
    --mode full

# 4b. Run XGBoost-only (fast baseline, ~2 min)
python src/train_predict.py \
    --train data/train.csv \
    --test  data/test.csv \
    --submission data/submission_format.csv \
    --output outputs/submission.csv \
    --mode xgb_only
```

---

## Architecture Overview

```
Raw GNSS channels (13 cols × N rows)
            │
            ▼
    ┌─────────────────┐
    │  Feature Engine  │  Physics-inspired features (Section 4)
    └────────┬────────┘
             │
    ┌────────┴─────────────────────────────┐
    │                                       │
    ▼                                       ▼
Timestamp-aggregated                Per-PRN Sequences
  tabular features                  (sliding window, len=20)
    │                                   │         │
    ▼                                   ▼         ▼
┌────────┐                       ┌────────┐  ┌───────────┐
│XGBoost │                       │BiLSTM  │  │Transformer│
│(5-fold)│                       │(BiDir) │  │(4 layers) │
└───┬────┘                       └───┬────┘  └─────┬─────┘
    │  OOF probs (timestamps)        │              │
    │                           map to timestamps   │
    └──────────────┬────────────────┘──────────────┘
                   ▼
         ┌──────────────────┐
         │  Meta-Learner    │  Logistic Regression on OOF probabilities
         │ (Stacking Layer) │
         └────────┬─────────┘
                  ▼
            Final Prediction
           (0 = Genuine, 1 = Spoofed)
```

---

## Feature Engineering

We build **physics-grounded** features that capture known spoofing signatures across 8 feature groups.

### 1. Correlator Distortion

```python
S_curve    = (EC - LC) / (EC + LC)   # ~0 for genuine; spoofed = asymmetric
peak_ratio = (EC + LC) / (2 * PC)    # ~1 for genuine; spoofed = shape distortion
EC_LC_ratio = EC / LC                # symmetry check
```

**Why it matters:** In a real GNSS correlator, the early (EC) and late (LC) correlator outputs are symmetric around the prompt (PC) peak. A spoofed signal broadcast from a different angle introduces code-tracking distortion — the correlation peak skews asymmetrically, measurable as a non-zero S-curve.

### 2. CN0 Anomaly (per-PRN z-score)

```python
CN0_zscore = (CN0 - per_PRN_mean_CN0) / per_PRN_std_CN0
CN0_high_flag = (CN0 > 47 dB-Hz)
CN0_time_deviation = CN0 - median(CN0 across all channels at same timestamp)
```

**Why it matters:** A spoofer must broadcast *stronger* than real satellites to overpower them. This shows up as an abnormally high CN0 relative to that satellite's historical baseline — often the single strongest discriminator in the dataset.

### 3. Temporal Rolling Features (per-PRN track)

```python
for col in [Doppler, Phase, Pseudorange, CN0, S_curve, ...]:
    col_diff   = per_PRN_first_difference(col)    # velocity of change
    col_diff2  = per_PRN_second_difference(col)   # acceleration — catches transition
    col_roll_std_{3,5,10,20} = rolling_std over windows
    col_roll_mean_{3,5,10,20} = rolling_mean over windows
```

**Why it matters:** Genuine satellite signals change **smoothly** — satellites move at predictable orbital speeds. Spoofed signals can jump abruptly when the attacker shifts their fake trajectory. The second-order difference (acceleration) is especially powerful at detecting the exact *start* and *end* of the attack.

### 4. Timing Mismatch

```python
time_diff = RX_time - TOW_at_current_symbol_s
```

**Why it matters:** The transmit time (TOW) and receiver timestamp should agree with the propagation delay. Spoofing introduces a subtle timing offset as the attacker replays a delayed signal from a different location.

### 5. Cross-Channel Doppler Deviation

```python
doppler_deviation     = |channel_Doppler - mean(Doppler across all channels)|
doppler_channel_std   = std(Doppler across all channels at same timestamp)
```

**Why it matters:** Genuine satellites produce Doppler shifts consistent with orbital geometry — they form a predictable spatial pattern. A spoofed channel breaks this geometric consensus, creating an outlier Doppler value.

### 6. Geometric Consistency (Pseudorange)

```python
pseudorange_spread    = std(Pseudorange across channels at same timestamp)
pseudorange_deviation = |channel_Pseudorange - mean(Pseudorange)|
```

**Why it matters:** A spoofing signal broadcasts from a *single* transmitter, so genuine multi-satellite geometry is disrupted. The spread of pseudoranges across satellites behaves differently under spoofing.

### 7. Channel Count

```python
channel_count = number of PRNs reporting at each timestamp
```

**Why it matters:** Spoofing can cause receivers to lose track of some real satellites while locking onto fakes, reducing channel count unexpectedly.

---

## Model Architecture: Three Base Learners

### 1. XGBoost (Tabular Expert)

Operates on **timestamp-level aggregated features** (`mean`, `std`, `max`, `min` across 8 channels).

| Parameter | Value | Reasoning |
|---|---|---|
| `n_estimators` | 700 | More trees, slower but better generalization |
| `max_depth` | 7 | Deep enough for feature interactions |
| `learning_rate` | 0.04 | Slower = more careful gradient steps |
| `subsample` | 0.8 | Row subsampling → diverse trees |
| `colsample_bytree` | 0.75 | Feature subsampling → robust features |
| `scale_pos_weight` | auto (neg/pos) | Corrects class imbalance |
| `gamma` | 0.1 | Minimum split gain → pruning overfitting |

**Evaluation:** 5-fold stratified CV, OOF weighted F1.

---

### 2. Bidirectional LSTM (Temporal Expert)

Operates on **per-PRN sliding windows** of length 20 timesteps.

```
Input (batch, 20, n_features)
  → BiLSTM(128) → LayerNorm → Dropout(0.3)
  → BiLSTM(64)  → LayerNorm → Dropout(0.3)
  → GlobalMaxPool (over time dimension)
  → Dense(64, GELU) → Dropout
  → Dense(1, sigmoid)
```

**Why Bidirectional?** The BiLSTM processes each sequence in both forward and backward directions, giving it context about both the approach to and recovery from a spoofing event — making it especially powerful at detecting transition boundaries.

**Why GlobalMaxPool?** Spoofing is a sustained block. The "worst" timestep in any window is most diagnostic — max pooling surfaces it.

**Training:** Weighted BCE loss (pos_weight = neg/pos), AdamW, Cosine LR annealing, early stopping on validation weighted F1.

---

### 3. Transformer Encoder (Cross-Timestep Attention Expert)

Operates on the same **per-PRN sliding windows** as LSTM.

```
Input (batch, 20, n_features)
  → Linear projection → d_model=128
  → Learnable positional encoding
  → 4x TransformerEncoderLayer (4 heads, ff=256, Pre-LN, GELU)
  → GlobalMaxPool + GlobalAvgPool → concat
  → Dense(128, GELU) → Dense(64, GELU) → Dense(1)
```

**Why Transformer?** Self-attention allows the model to directly compare any two timesteps in a window — learning that "timestep t12 shows the same Doppler anomaly as t8, and that's suspicious." It also captures long-range dependencies that BiLSTM may miss in longer windows.

**Why Pre-LayerNorm?** More training-stable than Post-LN for smaller datasets.

---

### 4. Meta-Learner (Stacking)

```
Stack: [XGB_prob, LSTM_prob, Transformer_prob] → Logistic Regression
```

All base model probabilities are generated as **Out-of-Fold (OOF)** predictions to prevent data leakage into the meta-learner. The Logistic Regression learns the optimal linear combination of the three models' outputs.

---

## Why Stacking is Better Than Simple Averaging

| Property | Simple Average | Stacking |
|---|---|---|
| Handles model confidence differences | ✗ | ✓ |
| Learns optimal blend weights | ✗ | ✓ |
| Adapts to model strengths per region | ✗ | ✓ |
| Data leakage risk | Low | Controlled via OOF |

---

## Evaluation

**Primary metric:** Weighted F1 Score

```
Weighted F1 = Σ (class_freq_i × F1_i)
```

Weighted F1 is appropriate here because:
- The dataset is imbalanced (~14% spoofed)
- Accuracy would reward always predicting "genuine"
- Weighted F1 forces the model to perform well on the minority (spoofed) class proportionally

**Cross-validation:** 5-fold stratified (same class ratio in each fold).

---

## Key Insights from EDA

**1. Class Imbalance (~14% spoofed)**
- Direct consequence: `scale_pos_weight ≈ 6` for XGBoost; `pos_weight` in BCE loss for deep models
- Weighted F1 ensures spoofed class is not buried

**2. Spoofing is One Continuous Block**
- The attack spans a contiguous range of timestamps (~50k–65k)
- This makes **temporal models (LSTM, Transformer) especially powerful** — they explicitly model the transition
- Rolling features with multiple window sizes (3, 5, 10, 20) capture the event at different temporal scales

**3. CN0 is the Strongest Discriminator**
- Spoofed signals cluster at CN0 ≈ 45–50 dB-Hz (unnaturally high)
- A spoofer must broadcast louder than the real satellite to overpower it
- Per-PRN z-score normalizes for genuine satellite power differences

**4. Raw Correlator Values are Near-Redundant**
- EC, LC, PC have ~0.99 pairwise correlation — raw values carry little information
- S-curve distortion and peak ratio capture the *shape* of the correlation peak, which is what actually changes under spoofing

**5. Geometric Consistency Breaks Under Spoofing**
- Cross-channel Doppler spread and pseudorange spread reflect satellite geometry
- A single-transmitter spoofer cannot simultaneously fake realistic geometry for 8 satellites

---

## Design Decisions

| Decision | Alternative Considered | Justification |
|---|---|---|
| Hybrid ensemble | XGBoost alone | Deep models capture temporal patterns tree models cannot |
| BiLSTM | Unidirectional LSTM | Bidirectional context improves transition detection |
| GlobalMaxPool | Last hidden state | Spoofing is sustained; worst-case timestep is most diagnostic |
| Pre-LN Transformer | Post-LN | Training stability on medium-sized datasets |
| OOF stacking | Hold-out | OOF uses all training data without leakage |
| Weighted BCE loss | SMOTE | No risk of overfitting to synthetic samples |
| Per-PRN rolling features | Global rolling | Spoofing signatures are PRN-track-specific |

---

## File Structure

```
gnss-antispoofing/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission_format.csv
├── outputs/
│   └── submission.csv
├── src/
│   ├── features.py           # Feature engineering
│   ├── xgb_model.py          # XGBoost detector
│   ├── lstm_model.py         # Bidirectional LSTM
│   ├── transformer_model.py  # Transformer encoder
│   ├── ensemble.py           # Hybrid stacking ensemble
│   └── train_predict.py      # Main entry point
├── requirements.txt
└── README.md
```
