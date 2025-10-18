"""
Generate synthetic speech-like signals for testing the pipeline without recordings.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import soundfile as sf

# Create dataset structure without importing audio_utils (to avoid sounddevice dependency)
def create_dataset_structure(base_dir="data"):
    """Create directory structure for storing audio dataset."""
    paths = {
        'base': base_dir,
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'database': os.path.join(base_dir, 'database'),
        'test': os.path.join(base_dir, 'test'),
        'results': os.path.join(base_dir, 'results')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def generate_synthetic_speech(duration=3.0, sample_rate=44100, fundamental_freq=150, word_type='hello'):
    """
    Generate synthetic speech-like signal.

    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate
        fundamental_freq: Fundamental frequency (Hz)
        word_type: Type of word pattern

    Returns:
        signal: Synthetic speech signal
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)

    # Initialize signal
    signal = np.zeros(n_samples)

    # Define different patterns for different words
    if word_type == 'hello':
        # "Hello" pattern: Two syllables with rising-falling intonation
        # Syllable 1: "Hel" (0.0 - 0.6s)
        # Syllable 2: "lo" (0.7 - 1.3s)

        # Syllable 1
        seg1_mask = (t >= 0.2) & (t < 0.7)
        f1 = fundamental_freq * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        signal[seg1_mask] = (
            0.8 * np.sin(2 * np.pi * f1[seg1_mask] * t[seg1_mask]) +
            0.3 * np.sin(2 * np.pi * 2 * f1[seg1_mask] * t[seg1_mask]) +
            0.15 * np.sin(2 * np.pi * 3 * f1[seg1_mask] * t[seg1_mask])
        )

        # Syllable 2
        seg2_mask = (t >= 0.8) & (t < 1.4)
        f2 = fundamental_freq * 0.8 * (1 + 0.15 * np.sin(2 * np.pi * 2 * t))
        signal[seg2_mask] = (
            0.7 * np.sin(2 * np.pi * f2[seg2_mask] * t[seg2_mask]) +
            0.25 * np.sin(2 * np.pi * 2 * f2[seg2_mask] * t[seg2_mask])
        )

    elif word_type == 'estimation':
        # "Estimation" pattern: Four syllables (es-ti-ma-tion)

        syllables = [
            (0.2, 0.5, fundamental_freq * 1.1),
            (0.6, 0.9, fundamental_freq * 1.2),
            (1.0, 1.3, fundamental_freq * 1.0),
            (1.4, 1.9, fundamental_freq * 0.9)
        ]

        for start, end, freq in syllables:
            mask = (t >= start) & (t < end)
            f = freq * (1 + 0.1 * np.sin(2 * np.pi * 5 * t))
            signal[mask] = (
                0.8 * np.sin(2 * np.pi * f[mask] * t[mask]) +
                0.3 * np.sin(2 * np.pi * 2 * f[mask] * t[mask]) +
                0.15 * np.sin(2 * np.pi * 3 * f[mask] * t[mask]) +
                0.1 * np.sin(2 * np.pi * 4 * f[mask] * t[mask])
            )

    elif word_type == 'oakland':
        # "Oakland" pattern: Two syllables (Oak-land)

        # Syllable 1: "Oak"
        seg1_mask = (t >= 0.2) & (t < 0.9)
        f1 = fundamental_freq * 0.95 * (1 + 0.25 * np.sin(2 * np.pi * 2 * t))
        signal[seg1_mask] = (
            0.9 * np.sin(2 * np.pi * f1[seg1_mask] * t[seg1_mask]) +
            0.35 * np.sin(2 * np.pi * 2 * f1[seg1_mask] * t[seg1_mask]) +
            0.2 * np.sin(2 * np.pi * 3 * f1[seg1_mask] * t[seg1_mask])
        )

        # Syllable 2: "land"
        seg2_mask = (t >= 1.0) & (t < 1.6)
        f2 = fundamental_freq * 1.05 * (1 + 0.15 * np.sin(2 * np.pi * 3 * t))
        signal[seg2_mask] = (
            0.75 * np.sin(2 * np.pi * f2[seg2_mask] * t[seg2_mask]) +
            0.3 * np.sin(2 * np.pi * 2 * f2[seg2_mask] * t[seg2_mask])
        )

    # Apply envelope to make it more speech-like
    envelope = np.ones_like(signal)

    # Attack and decay for each voiced region
    voiced_regions = np.abs(signal) > 0.01
    if np.any(voiced_regions):
        # Find transitions
        transitions = np.diff(voiced_regions.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0]

        # Apply attack/decay
        attack_samples = int(0.01 * sample_rate)
        decay_samples = int(0.02 * sample_rate)

        for start in starts:
            attack_end = min(start + attack_samples, len(envelope))
            envelope[start:attack_end] = np.linspace(0, 1, attack_end - start)

        for end in ends:
            decay_start = max(0, end - decay_samples)
            envelope[decay_start:end] = np.linspace(1, 0, end - decay_start)

    signal = signal * envelope

    # Add slight formant structure
    formant_noise = 0.05 * np.random.randn(n_samples)
    signal = signal + formant_noise * (np.abs(signal) > 0.01)

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal


def generate_person_variant(base_signal, variant_id, sample_rate=44100):
    """
    Generate a variant of the base signal to simulate different speakers.

    Args:
        base_signal: Base synthetic signal
        variant_id: Person variant ID (1, 2, 3)
        sample_rate: Sampling rate

    Returns:
        variant_signal: Modified signal
    """
    signal = base_signal.copy()

    # Different modifications for each person
    if variant_id == 1:
        # Person 1: Slightly higher pitch
        # Time-stretch and pitch-shift simulation
        stretch_factor = 0.95
        new_length = int(len(signal) * stretch_factor)
        indices = np.linspace(0, len(signal) - 1, new_length)
        signal = np.interp(indices, np.arange(len(signal)), signal)

        # Pad/truncate to original length
        if len(signal) < len(base_signal):
            signal = np.pad(signal, (0, len(base_signal) - len(signal)))
        else:
            signal = signal[:len(base_signal)]

        # Slight amplitude variation
        signal = signal * 0.95

    elif variant_id == 2:
        # Person 2: Normal (base)
        # Add slight timing variation
        jitter = 0.02 * np.random.randn(len(signal))
        signal = signal + jitter * (np.abs(signal) > 0.01)

    elif variant_id == 3:
        # Person 3: Slightly lower pitch, slower
        stretch_factor = 1.05
        new_length = int(len(signal) * stretch_factor)
        indices = np.linspace(0, len(signal) - 1, new_length)
        signal = np.interp(indices, np.arange(len(signal)), signal)

        if len(signal) < len(base_signal):
            signal = np.pad(signal, (0, len(base_signal) - len(signal)))
        else:
            signal = signal[:len(base_signal)]

        signal = signal * 1.05

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal


def main():
    """
    Generate synthetic dataset.
    """
    print("=" * 60)
    print("GENERATING SYNTHETIC SPEECH DATA")
    print("=" * 60)

    # Parameters
    sample_rate = 44100
    duration = 3.0
    words = ['Hello', 'Estimation', 'Oakland']
    num_people = 3

    # Create directory structure
    paths = create_dataset_structure()

    # Generate signals
    for word in words:
        print(f"\nGenerating '{word}'...")

        # Generate base signal
        base_signal = generate_synthetic_speech(
            duration=duration,
            sample_rate=sample_rate,
            fundamental_freq=150,
            word_type=word.lower()
        )

        # Generate variants for different people
        for person_id in range(1, num_people + 1):
            signal = generate_person_variant(base_signal, person_id, sample_rate)

            # Save
            filename = os.path.join(paths['raw'], f"{word}_person{person_id}.wav")
            sf.write(filename, signal, sample_rate)
            print(f"  Saved: {filename}")

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Files saved in: {paths['raw']}")


if __name__ == "__main__":
    main()
