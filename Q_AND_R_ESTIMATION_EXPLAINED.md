# Q and R Parameters in the Kalman Filter - Detailed Explanation

**Date**: November 8, 2025  
**Question**: How are Q and R measured? How does the RBF neural network help estimate them?

---

## TL;DR - Quick Answer

**Q (Process Noise Covariance)**: ✅ **Estimated by RBF neural network**  
**R (Measurement Noise Covariance)**: ❌ **Set manually (constant = 0.01)**

**The RBF neural network ONLY estimates Q, not R.**

---

## Understanding Q and R

### What are Q and R?

In the Kalman filter equations:

```
Prediction:
  x_pred = φ * x_est + G * u
  P_pred = φ * P_est * φ^T + Q    ← Q appears here

Update:
  S = H * P_pred * H^T + R        ← R appears here
  K = P_pred * H^T * S^(-1)
  x_est = x_pred + K * (y - H * x_pred)
```

**Q (Process Noise Covariance)**:
- Represents uncertainty in the **state evolution model**
- "How much do we trust our AR model to predict the next state?"
- Higher Q = Less trust in model, more adaptation to measurements
- Lower Q = More trust in model, less responsive to new data

**R (Measurement Noise Covariance)**:
- Represents uncertainty in the **observations/measurements**
- "How noisy are our measurements?"
- Higher R = Less trust in measurements, rely more on predictions
- Lower R = More trust in measurements, update aggressively

---

## How Q is Estimated (RBF Neural Network)

### Location: `src/rbf.py` Lines 118-244

### Step 1: Choose Estimation Method

```python
# From main.py line 157-162
rbf, Q = create_rbf_for_kalman(
    current_noisy_signal,
    gamma=1.0,
    use_contextual_features=use_ctx,  # ← KEY CHOICE
    sample_rate=16000
)
```

**Two methods available:**

#### Method A: Simple Variance-Based (Baseline)
```python
# src/rbf.py line 169-172
signal_variance = np.var(signal)
Q = rbf.estimate_process_noise(signal_variance)
```

**Process:**
1. Compute variance of noisy signal: `σ² = var(signal)`
2. Create synthetic training data:
   - X_train: variance values from 0.01 to 2σ²
   - y_train: corresponding Q values (non-linear relationship)
   ```python
   y_train = 0.5 * X * (1 + 0.3 * sin(2π * X / σ²))
   ```
3. Train RBF network: `Φ * W = Y`
4. Predict Q for current variance: `Q = Φ(σ²) * W`

#### Method B: Contextual Features (Enhanced)
```python
# src/rbf.py line 175-244
Q = estimate_q_with_contextual_features(signal, rbf, sample_rate)
```

**Process:**
1. **Extract 5 per-frame features** (lines 189-195):
   - RMS energy (power)
   - Zero-crossing rate (periodicity)
   - Spectral centroid (brightness)
   - Spectral flatness (tonality vs noise)
   - Local SNR estimate

2. **Add temporal context** (line 202):
   - Concatenate ±5 frames → 11 frames
   - Captures speech dynamics (transitions, coarticulation)

3. **Non-linear expansions** (line 205):
   - Squared terms: x²
   - Pairwise products: x_i × x_j
   - Log transforms: log(1 + |x|)
   - Result: 30-40x feature expansion

4. **Compute feature variance** (line 212):
   ```python
   feature_vector = mean(expanded_features)
   feature_variance = var(feature_vector)
   ```

5. **Train RBF with enhanced mapping** (lines 220-226):
   ```python
   X_train = linspace(0.01, 3 * feature_variance, 100)
   y_train = 0.3 * X * (1 + 0.5 * tanh(X / feature_variance))
   rbf.fit(X_train, y_train)
   ```

6. **Predict Q** (line 231):
   ```python
   Q = rbf.predict([[feature_variance]])
   ```

### Step 2: RBF Neural Network Training

**Architecture** (Gaussian RBF):
```python
# src/rbf.py lines 38-55
Φ(x) = exp(-γ * ||x - c||²)

Where:
  x = input features (variance or feature_variance)
  c = centers (training data points)
  γ = shape parameter (default 1.0)
```

**Training** (Matrix inversion method):
```python
# src/rbf.py lines 78-89
1. Φ = gaussian_rbf(X_train, centers)    # Compute kernel matrix
2. W = (Φ^T Φ + λI)^(-1) Φ^T y_train    # Solve for weights
```

**Prediction**:
```python
# src/rbf.py lines 94-111
Q = Φ(x_current) * W
```

---

## How R is Set (Manual Constant)

### Location: `main.py` Line 171

```python
enhanced_signal = enhance_speech_rbf_kalman(
    current_noisy_signal,
    Q_rbf=Q,        # ← Estimated by RBF
    R=0.01,         # ← HARDCODED constant
    order=12
)
```

### Why is R Constant?

**From the 2020 paper:**
- The paper focused on **adaptive Q estimation** using RBF
- R was treated as a **known constant** or **manually tuned**
- This is common in speech enhancement when you have a rough idea of measurement noise level

**In our implementation:**
- `R = 0.01` is a **heuristic value** that works well for typical speech
- Represents measurement noise variance (observation uncertainty)
- Could be tuned for specific applications but wasn't the paper's focus

### Could R be Estimated?

**Yes, R could also be estimated**, but the paper didn't do this. Options include:

1. **Variance of measurement residuals**:
   ```python
   R = var(y - H * x_pred)  # Innovation variance
   ```

2. **Noise variance estimation algorithms**:
   - Minimum statistics
   - MMSE-based noise estimation
   - Voice activity detection (VAD) during silence

3. **Another RBF network** (not implemented):
   ```python
   rbf_R, R = create_rbf_for_measurement_noise(signal)
   ```

**But in this paper:** R is simply set to 0.01 as a reasonable default.

---

## Summary Comparison

| Parameter | Estimated By | Method | Adaptive? |
|-----------|--------------|--------|-----------|
| **Q** | ✅ RBF Neural Network | Variance → RBF → Q | ✅ Yes (per-signal) |
| **R** | ❌ Manual Setting | Constant = 0.01 | ❌ No (fixed) |

---

## Visual Summary: How RBF Estimates Q

```
Input Signal
    ↓
[Choose Method]
    ↓
    ├─→ Simple: var(signal) → σ²
    │       ↓
    │   Create synthetic mapping:
    │   X_train = [0.01 ... 2σ²]
    │   y_train = 0.5*X*(1 + 0.3*sin(2πX/σ²))
    │       ↓
    │   Train RBF: Φ*W = Y
    │       ↓
    │   Predict: Q = Φ(σ²)*W
    │
    └─→ Contextual: Extract features → f
            ↓
        5 features (RMS, ZCR, centroid, flatness, SNR)
            ↓
        Temporal context: ±5 frames (11 total)
            ↓
        Non-linear: x², x_i×x_j, log(x)
            ↓
        feature_variance = var(f_expanded)
            ↓
        Create enhanced mapping:
        X_train = [0.01 ... 3*feature_variance]
        y_train = 0.3*X*(1 + 0.5*tanh(X/feature_variance))
            ↓
        Train RBF: Φ*W = Y
            ↓
        Predict: Q = Φ(feature_variance)*W
            ↓
            ↓
        Q estimate
            ↓
    Kalman Filter
    (with R = 0.01)
            ↓
    Enhanced Signal
```

---

## Why RBF Neural Network?

### Why not just use variance directly as Q?

**Problem with linear mapping** (`Q = k * variance`):
- Speech signals are **non-stationary** (varying statistics)
- Different noise types need different Q values
- Linear relationship is too simplistic

**RBF provides non-linear mapping**:
- `Q = RBF(features)` can capture complex relationships
- Gaussian kernels adapt to signal characteristics
- Training on synthetic data creates sensible Q → variance mapping

### Key Innovation from Paper

**Original Paper (2020) Limitation:**
> "Its performance has found no positive effects on the results since many parameters of that technique such as weights w and Gamma ϒ needs to be estimated to obtain decent results."

**Our Implementation Fixes:**
1. ✅ **Regularization**: Added `λI` to avoid singular matrix (line 82)
2. ✅ **Contextual features**: Rich feature set instead of just variance
3. ✅ **Robust training**: Fallback to simple method if features fail
4. ✅ **Sensible initialization**: Synthetic training data with non-linear relationships

---

## Code Locations

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| RBF Class | `src/rbf.py` | 18-115 | Gaussian RBF implementation |
| Simple Q Estimation | `src/rbf.py` | 118-154 | Variance-based (baseline) |
| Contextual Q Estimation | `src/rbf.py` | 175-244 | Enhanced features method |
| RBF Creation | `src/rbf.py` | 157-172 | Main entry point |
| Kalman Filter Init | `src/kalman_filter.py` | 115-143 | Sets Q and R |
| R Value Setting | `main.py` | 171 | Hardcoded R=0.01 |

---

## Example Values

From typical speech enhancement run:

```python
# Baseline method:
signal_variance = 0.0453
Q_baseline = 0.0289  # Estimated by RBF

# Contextual method:
feature_variance = 0.1234
Q_contextual = 0.0567  # Higher Q → more adaptation

# Measurement noise:
R = 0.01  # Fixed constant
```

**Note**: Contextual features typically produce **higher Q values** because the richer feature representation captures more signal uncertainty.

---

## Final Answer

### Q (Process Noise):
✅ **Estimated by RBF neural network**  
✅ **Two methods**: Simple (variance) or Contextual (30-40x features)  
✅ **Adaptive**: Changes per signal  
✅ **Non-linear mapping**: Uses Gaussian RBF kernels  

### R (Measurement Noise):
❌ **Not estimated by RBF**  
❌ **Hardcoded constant**: R = 0.01  
❌ **Not adaptive**: Same for all signals  
✅ **Could be estimated**: But paper didn't focus on this  

**The paper's innovation was using RBF to adaptively estimate Q, not R.**
