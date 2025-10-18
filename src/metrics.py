"""
Advanced audio quality metrics for speech enhancement evaluation.

Implements:
- Segmental SNR
- Signal-to-Distortion Ratio (SDR)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)

Based on ChatGPT requirements for comprehensive evaluation.
"""

import numpy as np
from scipy import signal as sp_signal


def segmental_snr(clean, enhanced, frame_len=160, eps=1e-10):
    """
    Compute Segmental Signal-to-Noise Ratio.

    Averages SNR over short frames (typically 10-20ms).
    More sensitive to local quality than global SNR.

    Args:
        clean: Reference clean signal
        enhanced: Enhanced/processed signal
        frame_len: Frame length in samples (e.g., 160 @ 16kHz = 10ms)
        eps: Small constant to avoid log(0)

    Returns:
        seg_snr: Segmental SNR in dB
    """
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # Number of complete frames
    n_frames = min_len // frame_len

    snr_frames = []

    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len

        clean_frame = clean[start:end]
        enhanced_frame = enhanced[start:end]

        # Signal power
        signal_power = np.sum(clean_frame ** 2)

        # Noise power (distortion)
        noise_power = np.sum((clean_frame - enhanced_frame) ** 2)

        # SNR for this frame (in dB)
        if noise_power > eps and signal_power > eps:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 100.0  # Perfect if no noise

        # Clip extreme values
        snr_db = np.clip(snr_db, -20, 100)

        snr_frames.append(snr_db)

    # Average over frames
    seg_snr = np.mean(snr_frames) if snr_frames else 0.0

    return seg_snr


def signal_to_distortion_ratio(clean, enhanced):
    """
    Compute Signal-to-Distortion Ratio (SDR).

    Measures overall signal quality.

    Args:
        clean: Reference clean signal
        enhanced: Enhanced signal

    Returns:
        sdr: SDR in dB
    """
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # Signal power
    signal_power = np.sum(clean ** 2)

    # Distortion power
    distortion_power = np.sum((clean - enhanced) ** 2)

    if distortion_power > 1e-10:
        sdr = 10 * np.log10(signal_power / distortion_power)
    else:
        sdr = 100.0

    return sdr


def compute_pesq(clean, enhanced, sr=16000, mode='wb'):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).

    Requires pesq library: pip install pesq

    Args:
        clean: Reference clean signal
        enhanced: Enhanced signal
        sr: Sample rate (8000 or 16000)
        mode: 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz)

    Returns:
        pesq_score: PESQ score (higher is better, range ~1-5)
    """
    try:
        from pesq import pesq as pesq_func

        # Ensure correct sample rate
        if sr not in [8000, 16000]:
            raise ValueError("PESQ requires sr=8000 or 16000")

        # Ensure same length and float32
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len].astype(np.float32)
        enhanced = enhanced[:min_len].astype(np.float32)

        # Normalize to [-1, 1]
        clean = clean / (np.max(np.abs(clean)) + 1e-10)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-10)

        # Compute PESQ
        pesq_score = pesq_func(sr, clean, enhanced, mode)

        return pesq_score

    except ImportError:
        print("Warning: pesq library not installed. Install with: pip install pesq")
        return None
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        return None


def compute_stoi(clean, enhanced, sr=16000):
    """
    Compute STOI (Short-Time Objective Intelligibility).

    Requires pystoi library: pip install pystoi

    Args:
        clean: Reference clean signal
        enhanced: Enhanced signal
        sr: Sample rate

    Returns:
        stoi_score: STOI score (0-1, higher is better)
    """
    try:
        from pystoi import stoi as stoi_func

        # Ensure same length and float
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len].astype(float)
        enhanced = enhanced[:min_len].astype(float)

        # Normalize
        clean = clean / (np.max(np.abs(clean)) + 1e-10)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-10)

        # Compute STOI
        stoi_score = stoi_func(clean, enhanced, sr, extended=False)

        return stoi_score

    except ImportError:
        print("Warning: pystoi library not installed. Install with: pip install pystoi")
        return None
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return None


def compute_all_metrics(clean, enhanced, sr=16000):
    """
    Compute all available metrics.

    Args:
        clean: Reference clean signal
        enhanced: Enhanced signal
        sr: Sample rate

    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {}

    # Always compute these
    metrics['seg_snr'] = segmental_snr(clean, enhanced, frame_len=int(sr*0.01))
    metrics['sdr'] = signal_to_distortion_ratio(clean, enhanced)

    # Global SNR for comparison
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - enhanced) ** 2)
    if noise_power > 1e-10:
        metrics['snr'] = 10 * np.log10(signal_power / noise_power)
    else:
        metrics['snr'] = 100.0

    # MSE
    metrics['mse'] = np.mean((clean - enhanced) ** 2)

    # Optional: PESQ and STOI (if libraries installed)
    if sr in [8000, 16000]:
        pesq_score = compute_pesq(clean, enhanced, sr)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score

    stoi_score = compute_stoi(clean, enhanced, sr)
    if stoi_score is not None:
        metrics['stoi'] = stoi_score

    return metrics


def print_metrics(metrics, title="Metrics"):
    """
    Pretty-print metrics dictionary.

    Args:
        metrics: Dictionary from compute_all_metrics()
        title: Title for output
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if value is not None:
            if 'snr' in key.lower() or 'sdr' in key.lower():
                print(f"  {key.upper():12s}: {value:8.2f} dB")
            elif key == 'mse':
                print(f"  {key.upper():12s}: {value:8.6f}")
            else:
                print(f"  {key.upper():12s}: {value:8.4f}")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics with synthetic signal
    print("Testing metrics module...")

    # Generate clean signal
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    clean = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Add noise
    noise = 0.1 * np.random.randn(len(clean))
    noisy = clean + noise

    # Simple "enhancement" (attenuate)
    enhanced = 0.9 * clean + 0.05 * noise

    # Compute metrics
    print("\nComparing: Noisy vs Enhanced")

    print("\n--- Noisy Signal ---")
    metrics_noisy = compute_all_metrics(clean, noisy, sr)
    print_metrics(metrics_noisy, "Noisy Signal Metrics")

    print("\n--- Enhanced Signal ---")
    metrics_enhanced = compute_all_metrics(clean, enhanced, sr)
    print_metrics(metrics_enhanced, "Enhanced Signal Metrics")

    # Improvement
    print("\n--- Improvement ---")
    print(f"  SegSNR improvement: {metrics_enhanced['seg_snr'] - metrics_noisy['seg_snr']:.2f} dB")
    print(f"  SDR improvement:    {metrics_enhanced['sdr'] - metrics_noisy['sdr']:.2f} dB")

    if 'pesq' in metrics_enhanced:
        print(f"  PESQ improvement:   {metrics_enhanced['pesq'] - metrics_noisy.get('pesq', 0):.2f}")

    if 'stoi' in metrics_enhanced:
        print(f"  STOI improvement:   {metrics_enhanced['stoi'] - metrics_noisy.get('stoi', 0):.4f}")

    print("\n✓ Metrics module test complete")
