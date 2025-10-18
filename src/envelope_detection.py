"""
Signal envelope detection for removing silent portions.
Based on the paper: Speech Enhancement and Recognition using Kalman Filter Modified via RBF
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt


class EnvelopeDetector:
    """
    Detect and extract the envelope of audio signals to separate voiced and silent regions.
    """

    def __init__(self, threshold_ratio=0.1, window_size=256):
        """
        Initialize envelope detector.

        Args:
            threshold_ratio: Ratio of maximum amplitude to use as threshold
            window_size: Window size for envelope smoothing
        """
        self.threshold_ratio = threshold_ratio
        self.window_size = window_size

    def compute_envelope(self, signal):
        """
        Compute signal envelope using Hilbert transform.

        Args:
            signal: Input signal

        Returns:
            envelope: Signal envelope (amplitude)
        """
        # Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert(signal)

        # Envelope is the magnitude of the analytic signal
        envelope = np.abs(analytic_signal)

        # Smooth the envelope using moving average
        envelope = self._smooth_envelope(envelope)

        return envelope

    def _smooth_envelope(self, envelope):
        """
        Smooth envelope using moving average filter.

        Args:
            envelope: Raw envelope

        Returns:
            smoothed_envelope: Smoothed envelope
        """
        # Create moving average kernel
        kernel = np.ones(self.window_size) / self.window_size

        # Apply convolution for smoothing
        smoothed_envelope = np.convolve(envelope, kernel, mode='same')

        return smoothed_envelope

    def detect_voiced_regions(self, signal, envelope=None):
        """
        Detect voiced (non-silent) regions using envelope threshold.

        Args:
            signal: Input signal
            envelope: Pre-computed envelope (optional)

        Returns:
            voiced_mask: Boolean mask (True for voiced regions)
        """
        if envelope is None:
            envelope = self.compute_envelope(signal)

        # Determine threshold
        threshold = self.threshold_ratio * np.max(envelope)

        # Create voiced mask
        voiced_mask = envelope > threshold

        return voiced_mask

    def extract_voiced_signal(self, signal):
        """
        Extract only the voiced portions of the signal.

        Args:
            signal: Input signal

        Returns:
            voiced_signal: Signal with only voiced portions
            voiced_indices: Indices of voiced samples
        """
        # Compute envelope
        envelope = self.compute_envelope(signal)

        # Detect voiced regions
        voiced_mask = self.detect_voiced_regions(signal, envelope)

        # Extract voiced samples
        voiced_indices = np.where(voiced_mask)[0]
        voiced_signal = signal[voiced_mask]

        return voiced_signal, voiced_indices

    def remove_silence(self, signal, return_indices=False):
        """
        Remove silent portions from signal.

        Args:
            signal: Input signal
            return_indices: Whether to return indices of voiced samples

        Returns:
            voiced_signal: Signal without silence
            voiced_indices: Indices of voiced samples (if return_indices=True)
        """
        voiced_signal, voiced_indices = self.extract_voiced_signal(signal)

        if return_indices:
            return voiced_signal, voiced_indices
        else:
            return voiced_signal


def detect_envelope(signal, threshold_ratio=0.1):
    """
    Convenience function for envelope detection.

    Args:
        signal: Input signal
        threshold_ratio: Threshold ratio for voiced detection

    Returns:
        envelope: Signal envelope
        voiced_mask: Boolean mask for voiced regions
    """
    detector = EnvelopeDetector(threshold_ratio=threshold_ratio)
    envelope = detector.compute_envelope(signal)
    voiced_mask = detector.detect_voiced_regions(signal, envelope)

    return envelope, voiced_mask


def remove_silence_from_signal(signal, threshold_ratio=0.1):
    """
    Remove silence from audio signal.

    Args:
        signal: Input signal
        threshold_ratio: Threshold ratio for silence detection

    Returns:
        voiced_signal: Signal without silent portions
    """
    detector = EnvelopeDetector(threshold_ratio=threshold_ratio)
    voiced_signal = detector.remove_silence(signal)

    return voiced_signal
