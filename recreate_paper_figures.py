"""
Recreate all figures from the paper exactly as shown.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visualization import SpeechVisualizer
from rbf import create_rbf_for_kalman
from kalman_filter import enhance_speech_rbf_kalman
from envelope_detection import EnvelopeDetector
import soundfile as sf

print("=" * 60)
print("RECREATING PAPER FIGURES")
print("=" * 60)

visualizer = SpeechVisualizer()
sample_rate = 44100

# Load synthetic data
data_dir = "data/raw"
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)

# Figure 1: Time Domain of "Estimation" from 3 people
print("\n--- Figure 1: Time Domain Estimation Signals ---")
estimation_signals = []
estimation_titles = []

for i in range(1, 4):
    filename = os.path.join(data_dir, f"Estimation_person{i}.wav")
    try:
        audio, sr = sf.read(filename)
        if sr != sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        estimation_signals.append(audio)
        if i == 1:
            estimation_titles.append('Mario Says "Estimation"')
        elif i == 2:
            estimation_titles.append('Farag Says "Estimation"')
        else:
            estimation_titles.append('Amr Says "Estimation"')
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if estimation_signals:
    visualizer.plot_time_domain(
        estimation_signals,
        estimation_titles,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure1_Time_Domain_Estimation.png')
    )
    print("✓ Generated Figure 1")

# Figure 2: Frequency Domain of "Estimation" from 3 people
print("\n--- Figure 2: Frequency Domain Estimation Signals ---")
if estimation_signals:
    freq_titles = [
        'Single-sided Magnitude spectrum (Hertz) of Voice of Mario',
        'Single-sided Magnitude spectrum (Hertz) of Voice of Farag',
        'Single-sided Magnitude spectrum (Hertz) of Voice of Amr'
    ]
    visualizer.plot_frequency_domain(
        estimation_signals,
        freq_titles,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure2_Frequency_Domain_Estimation.png')
    )
    print("✓ Generated Figure 2")

# Figure 3: Test Signals Time Domain (Hello, Estimation, Oakland)
print("\n--- Figure 3: Test Signals Time Domain ---")
test_words = ['Hello', 'Estimation', 'Oakland']
test_signals = []
test_titles = []

for word in test_words:
    filename = os.path.join(data_dir, f"{word}_person1.wav")
    try:
        audio, sr = sf.read(filename)
        if sr != sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        test_signals.append(audio)
        test_titles.append(f'A Test Signal Says "{word}"')
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if test_signals:
    visualizer.plot_time_domain(
        test_signals,
        test_titles,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure3_Test_Signals_Time.png')
    )
    print("✓ Generated Figure 3")

# Figure 4: Test Signals Frequency Domain
print("\n--- Figure 4: Test Signals Frequency Domain ---")
if test_signals:
    freq_test_titles = [f'Single-sided Magnitude spectrum (Hertz) of Voice of a Test Signal' for _ in test_words]
    visualizer.plot_frequency_domain(
        test_signals,
        freq_test_titles,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure4_Test_Signals_Frequency.png')
    )
    print("✓ Generated Figure 4")

# Figure 5: Envelope Detection
print("\n--- Figure 5: Envelope Detection ---")
detector = EnvelopeDetector()

# Use first estimation signal
if estimation_signals:
    signal = estimation_signals[0]
    envelope = detector.compute_envelope(signal)
    voiced_mask = detector.detect_voiced_regions(signal, envelope)

    visualizer.plot_envelope_detection(
        signal,
        envelope,
        voiced_mask,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure5_Envelope_Detection.png')
    )
    print("✓ Generated Figure 5")

# Figure 6: Voiced Signal (Edited Signal)
print("\n--- Figure 6: Voiced Signal ---")
if estimation_signals:
    voiced_signal = detector.remove_silence(signal)

    visualizer.plot_voiced_signal(
        voiced_signal,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure6_Voiced_Signal.png')
    )
    print("✓ Generated Figure 6")

# Figure 7: Enhancement Results
print("\n--- Figure 7: RBF-Kalman Enhancement Results ---")
if estimation_signals:
    clean_signal = estimation_signals[0]

    # Add noise
    signal_power = np.mean(clean_signal ** 2)
    snr_db = 10
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise

    # Enhance using RBF-Kalman
    rbf, Q = create_rbf_for_kalman(noisy_signal)
    enhanced_signal = enhance_speech_rbf_kalman(noisy_signal, Q, R=0.01, order=10)

    # Normalize for plotting
    max_val = max(np.max(np.abs(clean_signal)), np.max(np.abs(noisy_signal)), np.max(np.abs(enhanced_signal)))
    clean_norm = clean_signal / max_val
    noisy_norm = noisy_signal / max_val
    enhanced_norm = enhanced_signal / max_val

    visualizer.plot_enhancement_results(
        clean_norm,
        noisy_norm,
        enhanced_norm,
        sample_rate,
        save_path=os.path.join(results_dir, 'Figure7_Enhancement_Results.png')
    )
    print("✓ Generated Figure 7")

print("\n" + "=" * 60)
print("ALL PAPER FIGURES RECREATED")
print("=" * 60)
print(f"Results saved in: {results_dir}/")
print("\nGenerated Figures:")
print("- Figure1_Time_Domain_Estimation.png")
print("- Figure2_Frequency_Domain_Estimation.png")
print("- Figure3_Test_Signals_Time.png")
print("- Figure4_Test_Signals_Frequency.png")
print("- Figure5_Envelope_Detection.png")
print("- Figure6_Voiced_Signal.png")
print("- Figure7_Enhancement_Results.png")
print("=" * 60)
