"""
Run comprehensive experiments comparing baseline and enhanced methods.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rbf import create_rbf_for_kalman
from kalman_filter import enhance_speech_rbf_kalman
from enhanced_methods import (
    AdaptiveKalmanFilter,
    SpectralSubtraction,
    enhance_with_ensemble,
    MFCCFeatureExtractor
)
from speech_recognition import CorrelationSpeechRecognizer, DTWSpeechRecognizer
from envelope_detection import remove_silence_from_signal
from visualization import SpeechVisualizer
import soundfile as sf

# Minimal implementations to avoid sounddevice dependency
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

class AudioProcessor:
    """Minimal audio processor for experiments."""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def load_audio(self, filename):
        """Load audio from file."""
        try:
            audio, sr = sf.read(filename)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None

    def normalize(self, audio):
        """Normalize audio to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def add_noise(self, audio, snr_db=10):
        """Add white Gaussian noise."""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(audio))
        return audio + noise, noise

    def resample(self, audio, original_sr, target_sr):
        """Resample audio (simple method)."""
        if original_sr == target_sr:
            return audio
        from scipy import signal as sp_signal
        num_samples = int(len(audio) * target_sr / original_sr)
        return sp_signal.resample(audio, num_samples)


class ExperimentRunner:
    """
    Run comprehensive experiments and comparisons.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize experiment runner.

        Args:
            sample_rate: Sampling rate
        """
        self.sample_rate = sample_rate
        self.processor = AudioProcessor(sample_rate)
        self.visualizer = SpeechVisualizer()
        self.paths = create_dataset_structure()

        # Results storage
        self.results = {
            'baseline': {},
            'adaptive': {},
            'spectral_sub': {},
            'ensemble': {}
        }

    def compute_snr(self, clean_signal, noisy_signal):
        """
        Compute Signal-to-Noise Ratio.

        Args:
            clean_signal: Clean reference signal
            noisy_signal: Noisy signal

        Returns:
            snr_db: SNR in dB
        """
        noise = noisy_signal - clean_signal
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0:
            snr = signal_power / noise_power
            snr_db = 10 * np.log10(snr)
        else:
            snr_db = np.inf

        return snr_db

    def compute_mse(self, signal1, signal2):
        """
        Compute Mean Squared Error.

        Args:
            signal1: First signal
            signal2: Second signal

        Returns:
            mse: Mean squared error
        """
        min_len = min(len(signal1), len(signal2))
        mse = np.mean((signal1[:min_len] - signal2[:min_len]) ** 2)
        return mse

    def evaluate_enhancement(self, clean, noisy, enhanced, method_name):
        """
        Evaluate enhancement quality.

        Args:
            clean: Clean reference signal
            noisy: Noisy signal
            enhanced: Enhanced signal
            method_name: Name of the method

        Returns:
            metrics: Dictionary of metrics
        """
        # SNR improvement
        snr_before = self.compute_snr(clean, noisy)
        snr_after = self.compute_snr(clean, enhanced)
        snr_improvement = snr_after - snr_before

        # MSE
        mse_before = self.compute_mse(clean, noisy)
        mse_after = self.compute_mse(clean, enhanced)
        mse_reduction = (mse_before - mse_after) / mse_before * 100

        metrics = {
            'method': method_name,
            'snr_before': snr_before,
            'snr_after': snr_after,
            'snr_improvement': snr_improvement,
            'mse_before': mse_before,
            'mse_after': mse_after,
            'mse_reduction_percent': mse_reduction
        }

        return metrics

    def run_baseline_method(self, noisy_signal, Q_rbf, R=0.01, order=10):
        """
        Run baseline RBF-Kalman method from the paper.

        Args:
            noisy_signal: Noisy input
            Q_rbf: RBF-estimated Q
            R: Measurement noise
            order: AR order

        Returns:
            enhanced_signal: Enhanced signal
        """
        enhanced = enhance_speech_rbf_kalman(noisy_signal, Q_rbf, R, order)
        return enhanced

    def run_adaptive_method(self, noisy_signal, Q_rbf, R=0.01, order=10):
        """
        Run adaptive Kalman filter method.

        Args:
            noisy_signal: Noisy input
            Q_rbf: Initial Q
            R: Initial R
            order: AR order

        Returns:
            enhanced_signal: Enhanced signal
        """
        adaptive_kf = AdaptiveKalmanFilter(order=order, adaptation_rate=0.01)
        enhanced = adaptive_kf.filter_signal(noisy_signal, Q_rbf, R)
        return enhanced

    def run_spectral_subtraction_method(self, noisy_signal, Q_rbf, R=0.01, order=10):
        """
        Run spectral subtraction + Kalman method.

        Args:
            noisy_signal: Noisy input
            Q_rbf: RBF-estimated Q
            R: Measurement noise
            order: AR order

        Returns:
            enhanced_signal: Enhanced signal
        """
        spec_sub = SpectralSubtraction()
        preprocessed = spec_sub.process(noisy_signal)
        enhanced = enhance_speech_rbf_kalman(preprocessed, Q_rbf, R, order)
        return enhanced

    def run_ensemble_method(self, noisy_signal, Q_rbf, R=0.01, order=10):
        """
        Run ensemble method.

        Args:
            noisy_signal: Noisy input
            Q_rbf: RBF-estimated Q
            R: Measurement noise
            order: AR order

        Returns:
            enhanced_signal: Enhanced signal
        """
        enhanced = enhance_with_ensemble(noisy_signal, Q_rbf, R, order, self.sample_rate)
        return enhanced

    def run_comparison_experiment(self, clean_signal, snr_db=10, order=10):
        """
        Run comparison of all methods on a single signal.

        Args:
            clean_signal: Clean reference signal
            snr_db: SNR for noise addition
            order: AR order

        Returns:
            results: Dictionary of results for all methods
        """
        # Add noise
        noisy_signal, _ = self.processor.add_noise(clean_signal, snr_db)

        # Estimate Q using RBF
        rbf, Q = create_rbf_for_kalman(noisy_signal)

        print(f"\nEstimated Q: {Q:.6f}")

        # Run all methods
        methods = {
            'baseline': self.run_baseline_method,
            'adaptive': self.run_adaptive_method,
            'spectral_sub': self.run_spectral_subtraction_method,
            'ensemble': self.run_ensemble_method
        }

        results = {}

        for method_name, method_func in methods.items():
            print(f"Running {method_name}...")

            enhanced = method_func(noisy_signal, Q, R=0.01, order=order)

            # Evaluate
            metrics = self.evaluate_enhancement(clean_signal, noisy_signal, enhanced, method_name)

            results[method_name] = {
                'enhanced_signal': enhanced,
                'metrics': metrics
            }

            # Print metrics
            print(f"  SNR improvement: {metrics['snr_improvement']:.2f} dB")
            print(f"  MSE reduction: {metrics['mse_reduction_percent']:.2f}%")

        return results, noisy_signal

    def load_test_signals(self):
        """
        Load test signals from database.

        Returns:
            signals: Dictionary of loaded signals
        """
        signals = {}
        raw_path = self.paths['raw']

        if not os.path.exists(raw_path):
            print(f"Directory not found: {raw_path}")
            return signals

        files = [f for f in os.listdir(raw_path) if f.endswith('.wav')]

        for file in files:
            filepath = os.path.join(raw_path, file)
            audio, sr = self.processor.load_audio(filepath)

            if audio is not None:
                if sr != self.sample_rate:
                    audio = self.processor.resample(audio, sr, self.sample_rate)

                audio = self.processor.normalize(audio)
                key = file.replace('.wav', '')
                signals[key] = audio

        return signals

    def visualize_comparison(self, clean, noisy, results, save_prefix="comparison"):
        """
        Visualize comparison of all methods.

        Args:
            clean: Clean signal
            noisy: Noisy signal
            results: Results dictionary
            save_prefix: Prefix for saved figures
        """
        # Prepare signals for plotting
        signals_dict = {
            'Clean': clean,
            'Noisy': noisy
        }

        for method_name, method_data in results.items():
            signals_dict[method_name.capitalize()] = method_data['enhanced_signal']

        # Plot comparison
        fig = self.visualizer.plot_comparison_grid(
            signals_dict,
            self.sample_rate,
            save_path=os.path.join(self.paths['results'], f'{save_prefix}_grid.png')
        )

        # Plot metrics
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        methods = list(results.keys())
        snr_improvements = [results[m]['metrics']['snr_improvement'] for m in methods]
        mse_reductions = [results[m]['metrics']['mse_reduction_percent'] for m in methods]

        # SNR improvement
        axes[0].bar(methods, snr_improvements, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0].set_ylabel('SNR Improvement (dB)')
        axes[0].set_title('SNR Improvement by Method')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # MSE reduction
        axes[1].bar(methods, mse_reductions, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[1].set_ylabel('MSE Reduction (%)')
        axes[1].set_title('MSE Reduction by Method')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.paths['results'], f'{save_prefix}_metrics.png'), dpi=300)
        print(f"Saved metrics plot: {save_prefix}_metrics.png")

    def run_full_evaluation(self, snr_levels=[5, 10, 15, 20]):
        """
        Run full evaluation across different SNR levels.

        Args:
            snr_levels: List of SNR levels to test
        """
        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60)

        # Load signals
        signals = self.load_test_signals()

        if not signals:
            print("No signals found. Please run generate_synthetic_data.py first.")
            return

        # Test on first signal
        test_key = list(signals.keys())[0]
        clean_signal = signals[test_key]

        print(f"\nTesting on: {test_key}")

        # Test across different SNR levels
        all_results = {}

        for snr in snr_levels:
            print(f"\n--- SNR = {snr} dB ---")

            results, noisy = self.run_comparison_experiment(clean_signal, snr_db=snr)

            all_results[snr] = results

            # Visualize
            self.visualize_comparison(
                clean_signal,
                noisy,
                results,
                save_prefix=f'comparison_snr{snr}'
            )

        # Create summary plot
        self.plot_snr_summary(all_results, snr_levels)

    def plot_snr_summary(self, all_results, snr_levels):
        """
        Plot summary of results across SNR levels.

        Args:
            all_results: Results for all SNR levels
            snr_levels: List of SNR levels
        """
        import matplotlib.pyplot as plt

        methods = ['baseline', 'adaptive', 'spectral_sub', 'ensemble']
        colors = ['blue', 'green', 'orange', 'red']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # SNR improvement vs input SNR
        for method, color in zip(methods, colors):
            improvements = [all_results[snr][method]['metrics']['snr_improvement']
                          for snr in snr_levels]
            axes[0].plot(snr_levels, improvements, marker='o', label=method.capitalize(),
                        color=color, linewidth=2)

        axes[0].set_xlabel('Input SNR (dB)')
        axes[0].set_ylabel('SNR Improvement (dB)')
        axes[0].set_title('SNR Improvement vs Input SNR')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MSE reduction vs input SNR
        for method, color in zip(methods, colors):
            reductions = [all_results[snr][method]['metrics']['mse_reduction_percent']
                         for snr in snr_levels]
            axes[1].plot(snr_levels, reductions, marker='o', label=method.capitalize(),
                        color=color, linewidth=2)

        axes[1].set_xlabel('Input SNR (dB)')
        axes[1].set_ylabel('MSE Reduction (%)')
        axes[1].set_title('MSE Reduction vs Input SNR')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.paths['results'], 'snr_summary.png'), dpi=300)
        print(f"\nSaved SNR summary plot")


def main():
    """
    Main execution.
    """
    print("=" * 60)
    print("COMPREHENSIVE EXPERIMENT RUNNER")
    print("=" * 60)

    runner = ExperimentRunner(sample_rate=44100)

    # Run full evaluation
    runner.run_full_evaluation(snr_levels=[5, 10, 15, 20])

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved in: {runner.paths['results']}")


if __name__ == "__main__":
    main()
