"""
Visualization utilities for plotting signals and results.
Recreates figures from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SpeechVisualizer:
    """
    Visualization tools for speech signals and enhancement results.
    """

    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_time_domain(self, signals, titles, sample_rate=44100, save_path=None):
        """
        Plot signals in time domain (recreates Figures 1 and 3 from paper).

        Args:
            signals: List of signals to plot
            titles: List of titles for each signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals))

        if n_signals == 1:
            axes = [axes]

        for idx, (signal, title) in enumerate(zip(signals, titles)):
            time = np.arange(len(signal)) / sample_rate

            axes[idx].plot(time, signal, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Time (Sec)')
            axes[idx].set_ylabel('Amplitude')
            axes[idx].set_title(title)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([-1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_frequency_domain(self, signals, titles, sample_rate=44100, save_path=None):
        """
        Plot signals in frequency domain (recreates Figures 2 and 4 from paper).

        Args:
            signals: List of signals to plot
            titles: List of titles for each signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals))

        if n_signals == 1:
            axes = [axes]

        for idx, (signal, title) in enumerate(zip(signals, titles)):
            # Compute FFT
            n = len(signal)
            fft_result = np.fft.fft(signal)
            magnitude = np.abs(fft_result[:n // 2])
            frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

            axes[idx].plot(frequencies, magnitude, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Frequency (Hz)')
            axes[idx].set_ylabel('Magnitude')
            axes[idx].set_title(f'Single-sided Magnitude spectrum (Hertz) of {title}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, sample_rate / 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_envelope_detection(self, signal, envelope, voiced_mask, sample_rate=44100, save_path=None):
        """
        Plot envelope detection results (recreates Figure 5 from paper).

        Args:
            signal: Original signal
            envelope: Detected envelope
            voiced_mask: Boolean mask for voiced regions
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        time = np.arange(len(signal)) / sample_rate

        # Plot original signal
        ax.plot(time, signal, 'b-', linewidth=0.5, label='Audio signal', alpha=0.7)

        # Plot envelope
        ax.plot(time, envelope, 'r-', linewidth=2, label='Envelope')

        ax.set_xlabel('Time (Sec)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Original Signal + Envelope of Mario')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_voiced_signal(self, voiced_signal, sample_rate=44100, save_path=None):
        """
        Plot extracted voiced signal (recreates Figure 6 from paper).

        Args:
            voiced_signal: Voiced-only signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        samples = np.arange(len(voiced_signal))

        ax.plot(samples, voiced_signal, 'b-', linewidth=0.5)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.set_title('Edited Signal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_enhancement_results(self, original, noisy, enhanced, sample_rate=44100, save_path=None):
        """
        Plot speech enhancement results (recreates Figure 7 from paper).

        Args:
            original: Original clean signal
            noisy: Noisy signal
            enhanced: Enhanced signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        time_orig = np.arange(len(original)) / sample_rate
        time_noisy = np.arange(len(noisy)) / sample_rate
        time_enhanced = np.arange(len(enhanced)) / sample_rate

        # Original signal
        axes[0].plot(time_orig, original, 'g-', linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('ORIGINAL SIGNAL')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-2, 2])

        # Noisy speech signal
        axes[1].plot(time_noisy, noisy, 'b-', linewidth=0.5)
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Noisy Speech Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-2, 2])

        # Enhanced signal
        axes[2].plot(time_enhanced, enhanced, 'g-', linewidth=0.5)
        axes[2].set_xlabel('Time (Sec)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('ESTIMATED SIGNAL')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-2, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_comparison_grid(self, signals_dict, sample_rate=44100, save_path=None):
        """
        Plot multiple signals in a grid layout.

        Args:
            signals_dict: Dictionary of {label: signal}
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals_dict)
        n_cols = 2
        n_rows = (n_signals + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for idx, (label, signal) in enumerate(signals_dict.items()):
            time = np.arange(len(signal)) / sample_rate

            axes[idx].plot(time, signal, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Time (Sec)')
            axes[idx].set_ylabel('Amplitude')
            axes[idx].set_title(label)
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_signals, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_recognition_results(self, test_signal, recognized_word, confidence, all_scores,
                                 sample_rate=44100, save_path=None):
        """
        Plot speech recognition results.

        Args:
            test_signal: Test signal
            recognized_word: Recognized word
            confidence: Recognition confidence
            all_scores: Scores for all words
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)

        # Plot test signal
        ax1 = fig.add_subplot(gs[0, :])
        time = np.arange(len(test_signal)) / sample_rate
        ax1.plot(time, test_signal, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (Sec)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Test Signal - Recognized: "{recognized_word}" (Confidence: {confidence:.3f})')
        ax1.grid(True, alpha=0.3)

        # Plot FFT
        ax2 = fig.add_subplot(gs[1, 0])
        n = len(test_signal)
        fft_result = np.fft.fft(test_signal)
        magnitude = np.abs(fft_result[:n // 2])
        frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
        ax2.plot(frequencies, magnitude, 'b-', linewidth=0.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Spectrum')
        ax2.grid(True, alpha=0.3)

        # Plot recognition scores
        ax3 = fig.add_subplot(gs[1, 1])
        words = list(all_scores.keys())
        scores = list(all_scores.values())
        colors = ['green' if w == recognized_word else 'blue' for w in words]

        ax3.barh(words, scores, color=colors, alpha=0.7)
        ax3.set_xlabel('Correlation Score')
        ax3.set_title('Recognition Scores')
        ax3.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    @staticmethod
    def show():
        """Show all plots."""
        plt.show()

    @staticmethod
    def close_all():
        """Close all plots."""
        plt.close('all')
