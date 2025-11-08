# Two Operating Modes Explained

**Date**: November 8, 2025  
**Feature**: Dual-mode operation for research and production use

---

## Overview

`main.py` now supports **two operating modes** to handle different use cases:

1. **Test Mode** (default) - For research, validation, and algorithm testing
2. **Enhance Mode** - For production use with real noisy audio

**Both modes run ALL capabilities** - the only difference is whether synthetic noise is added or not.

---

## Mode 1: Test Mode (Default)

### Purpose
- Research and validation
- Algorithm testing and comparison
- Reproducing paper results
- Systematic noise type evaluation

### Usage
```bash
python3 main.py
# OR explicitly:
python3 main.py --mode test
```

### What It Does
```
Clean audio (data/raw/*.wav)
    ↓
Add synthetic noise (4 types × 4 SNR levels)
    ↓
Enhance with RBF-Kalman (baseline + contextual)
    ↓
Compute metrics (SNR improvement, MSE, etc.)
    ↓
Generate visualizations
```

### Input Requirements
- **Clean** audio files in `data/raw/`
- No background noise expected
- Format: `.wav` (any sample rate, will be resampled)

### What Happens
1. Loads clean audio from `data/raw/`
2. **Adds synthetic noise** (white, fan, street, ambient)
3. Tests at 4 SNR levels (5, 10, 15, 20 dB)
4. Enhances with RBF-Kalman (2 methods: baseline + contextual features)
5. Compares performance with clean reference
6. Generates **32 test conditions** (4 noise × 4 SNR × 2 methods)

### Outputs
- `data/test/` - Noisy samples (16 files)
- `data/processed/` - Enhanced samples (32 files)
- `data/results/` - Visualizations and metrics
- Performance metrics with SNR improvement

### Example
```bash
# Generate synthetic test data first
python3 generate_synthetic_data.py

# Run validation experiments
python3 main.py --mode test
```

**Result**: Comprehensive validation showing algorithm performance across multiple noise types and SNR levels.

---

## Mode 2: Enhance Mode

### Purpose
- Production use
- Processing real-world recordings
- Cleaning up noisy audio
- Direct enhancement without testing overhead

### Usage
```bash
python3 main.py --mode enhance
```

### What It Does
```
Noisy audio (data/raw/*.wav)
    ↓
Enhance with RBF-Kalman (baseline + contextual)
    ↓
Save enhanced versions
    ↓
Generate visualizations (no SNR metrics)
```

### Input Requirements
- **Already-noisy** audio files in `data/raw/`
- Real recordings with background noise
- Format: `.wav` (any sample rate, will be resampled)

### What Happens
1. Loads **already-noisy** audio from `data/raw/`
2. **Skips synthetic noise generation** (audio is already noisy!)
3. Enhances directly with RBF-Kalman (2 methods: baseline + contextual features)
4. No SNR computation (no clean reference available)
5. Saves enhanced versions

### Outputs
- `data/test/` - Copy of input noisy audio
- `data/processed/` - Enhanced audio (2 files per input: baseline + contextual)
- `data/results/` - Visualizations
- No SNR metrics (no clean reference)

### Example
```bash
# Put your noisy recordings in data/raw/
cp ~/recordings/noisy_speech.wav data/raw/

# Enhance directly
python3 main.py --mode enhance
```

**Result**: Two enhanced versions of your audio (baseline and contextual features methods).

---

## Comparison Table

| Feature | Test Mode | Enhance Mode |
|---------|-----------|--------------|
| **Input** | Clean audio | Noisy audio |
| **Adds Noise** | ✅ Yes (4 types × 4 SNR) | ❌ No (already noisy) |
| **RBF-Kalman Enhancement** | ✅ Yes | ✅ Yes |
| **Baseline Method** | ✅ Yes | ✅ Yes |
| **Contextual Features** | ✅ Yes | ✅ Yes |
| **Multi-layer RBF Classifier** | ✅ Yes | ✅ Yes |
| **Speech Recognition** | ✅ Yes (3 methods) | ✅ Yes (3 methods) |
| **Visualizations** | ✅ Yes | ✅ Yes |
| **SNR Metrics** | ✅ Yes (has clean ref) | ❌ No (no clean ref) |
| **MSE Metrics** | ✅ Yes (vs clean) | ✅ Yes (vs input) |
| **Output Files** | 49 files | 3 files per input |
| **Test Conditions** | 32 conditions | 1 condition per file |

---

## ALL Capabilities Run in Both Modes ✅

### What BOTH Modes Do:

1. **RBF-Kalman Enhancement**
   - ✅ Levinson-Durbin AR coefficient estimation
   - ✅ DARE (Discrete Algebraic Riccati Equation) initialization
   - ✅ Companion-form state-space model
   - ✅ RBF network for Q estimation

2. **Feature Engineering**
   - ✅ Contextual features (5 types × 11 frames)
   - ✅ Non-linear expansions (squared, products, log)
   - ✅ MFCC extraction
   - ✅ 30-40x feature expansion

3. **Two Enhancement Methods**
   - ✅ Baseline (simple variance-based Q)
   - ✅ Contextual features (advanced Q estimation)

4. **Multi-Layer RBF Classifier**
   - ✅ 2-layer architecture (40 + 20 centers)
   - ✅ Training on enhanced speech
   - ✅ Classification accuracy

5. **Speech Recognition (3 Methods)**
   - ✅ Correlation-based (paper's method)
   - ✅ DTW (Dynamic Time Warping)
   - ✅ RBF neural network classifier

6. **Visualizations**
   - ✅ Waveform plots
   - ✅ Spectrograms
   - ✅ SNR improvement heatmaps (test mode only)
   - ✅ Method comparison plots

7. **Audio Saving**
   - ✅ Input audio saved
   - ✅ Enhanced audio saved (both methods)
   - ✅ Organized file structure

---

## The ONLY Difference

### Test Mode:
```python
# Line 138 in main.py
current_noisy_signal, _ = processor.add_noise(clean_signal, snr_db, noise_type=noise_type)
```
**Adds synthetic noise** to clean audio

### Enhance Mode:
```python
# Line 147 in main.py
current_noisy_signal = original_noisy_signal
```
**Uses the already-noisy audio** without adding more noise

**Everything else is identical!**

---

## Use Cases

### When to Use Test Mode
- ✅ Validating the algorithm
- ✅ Comparing noise types
- ✅ Testing different SNR levels
- ✅ Reproducing paper results
- ✅ Research and development
- ✅ You have clean audio samples

### When to Use Enhance Mode
- ✅ Processing real recordings
- ✅ Production deployment
- ✅ Cleaning up podcast audio
- ✅ Enhancing phone calls
- ✅ Processing field recordings
- ✅ You have noisy audio samples

---

## Examples

### Example 1: Research Validation (Test Mode)
```bash
# Step 1: Generate synthetic clean audio
python3 generate_synthetic_data.py

# Step 2: Run comprehensive validation
python3 main.py --mode test

# Result: 
# ✅ 32 test conditions
# ✅ SNR improvement metrics
# ✅ Noise type comparison plots
# ✅ Complete performance analysis
```

### Example 2: Clean Up Real Recording (Enhance Mode)
```bash
# Step 1: Add your noisy audio
cp ~/recording_with_noise.wav data/raw/

# Step 2: Enhance directly
python3 main.py --mode enhance

# Result:
# ✅ data/processed/enhanced_baseline_recording_with_noise.wav
# ✅ data/processed/enhanced_contextual_recording_with_noise.wav
# ✅ Visualizations showing before/after
```

### Example 3: Both Modes Comparison
```bash
# Test Mode: Clean → Add Noise → Enhance
echo "Test Mode:"
python3 main.py --mode test
# Outputs: 49 files (1 clean + 16 noisy + 32 enhanced)

# Enhance Mode: Noisy → Enhance
echo "Enhance Mode:"
python3 main.py --mode enhance
# Outputs: 3 files (1 input copy + 2 enhanced)
```

---

## Implementation Details

### Mode Selection (Lines 564-589)
```python
parser.add_argument(
    '--mode',
    type=str,
    default='test',
    choices=['test', 'enhance'],
    help='Operating mode: "test" for synthetic noise testing, "enhance" for real noisy audio'
)
```

### Mode-Aware Processing (Lines 92-114)
```python
if mode == 'test':
    clean_signal = signal
    noise_types = ['white', 'fan', 'street', 'ambient']
    snr_levels = [5, 10, 15, 20]
else:  # enhance mode
    original_noisy_signal = signal
    clean_signal = None
    noise_types = ['original']
    snr_levels = [0]  # Dummy
```

### Conditional Noise Addition (Lines 134-150)
```python
if mode == 'test':
    # Add synthetic noise
    current_noisy_signal, _ = processor.add_noise(clean_signal, snr_db, noise_type)
else:
    # Use original noisy signal
    current_noisy_signal = original_noisy_signal
```

---

## Summary

✅ **Both modes run ALL capabilities**  
✅ **Test mode**: Clean audio → Add noise → Enhance → Compare  
✅ **Enhance mode**: Noisy audio → Enhance → Output  
✅ **Same enhancement algorithms** (RBF-Kalman, contextual features)  
✅ **Same recognition methods** (Correlation, DTW, RBF-NN)  
✅ **Same visualizations**  
✅ **Only difference**: Synthetic noise generation step  

**You get the full power of the system in both modes!** 🚀
