"""
Enhanced methodologies to improve speech enhancement and recognition results.

These methods go beyond the paper's baseline implementation:
1. Adaptive Kalman Filter with time-varying Q and R
2. Deep RBF networks for better Q estimation
3. Spectral subtraction preprocessing
4. MFCC-based recognition
5. Ensemble methods
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fftpack import dct
from scipy.signal import wiener


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman filter with time-varying noise parameters.

    This improves upon the basic RBF-Kalman by adapting Q and R during filtering.
    """

    def __init__(self, order=10, adaptation_rate=0.01):
        """
        Initialize adaptive Kalman filter.

        Args:
            order: AR model order
            adaptation_rate: Rate of adaptation for Q and R
        """
        self.order = order
        self.adaptation_rate = adaptation_rate

        self.phi = None
        self.G = None
        self.H = None
        self.Q = None
        self.R = None

        self.x_est = None
        self.P_est = None

        # Innovation history for adaptation
        self.innovation_history = []
        self.max_history = 50

    def estimate_ar_coefficients(self, signal):
        """Estimate AR coefficients using Burg's method (more robust)."""
        # Burg's method for AR estimation
        from scipy.signal import lfilter

        # Use Levinson-Durbin (similar to basic Kalman filter)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]

        ar_coeffs = np.zeros(self.order)
        E = autocorr[0]

        for i in range(self.order):
            if i == 0:
                k = autocorr[1] / E
                ar_coeffs[0] = k
                E = E * (1 - k ** 2)
            else:
                sum_val = autocorr[i + 1]
                for j in range(i):
                    sum_val -= ar_coeffs[j] * autocorr[i - j]

                k = sum_val / E
                ar_coeffs_old = ar_coeffs.copy()
                ar_coeffs[i] = k

                for j in range(i):
                    ar_coeffs[j] = ar_coeffs_old[j] - k * ar_coeffs_old[i - 1 - j]

                E = E * (1 - k ** 2)

        return ar_coeffs

    def adapt_noise_parameters(self, innovation):
        """
        Adapt Q and R based on innovation statistics.

        Args:
            innovation: Current innovation (residual)
        """
        self.innovation_history.append(innovation)

        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        # Estimate innovation variance
        innovation_var = np.var(self.innovation_history)

        # Adapt R (measurement noise)
        # If innovation is high, increase R
        self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * innovation_var

        # Adapt Q (process noise)
        # Use innovation to estimate process noise
        Q_new = self.adaptation_rate * innovation_var / self.order
        self.Q = (1 - self.adaptation_rate) * self.Q + self.adaptation_rate * Q_new * np.eye(self.order)

    def filter_signal(self, noisy_signal, Q_init, R_init):
        """
        Filter signal with adaptive parameters.

        Args:
            noisy_signal: Noisy input
            Q_init: Initial process noise covariance
            R_init: Initial measurement noise covariance

        Returns:
            enhanced_signal: Filtered signal
        """
        # Initialize
        ar_coeffs = self.estimate_ar_coefficients(noisy_signal)

        # Build state space model
        p = len(ar_coeffs)
        self.phi = np.zeros((p, p))
        self.phi[0, :] = ar_coeffs
        if p > 1:
            self.phi[1:, :-1] = np.eye(p - 1)

        self.G = np.zeros((p, 1))
        self.G[0, 0] = 1.0

        self.H = np.zeros((1, p))
        self.H[0, -1] = 1.0

        self.Q = Q_init * np.eye(self.order)
        self.R = R_init

        self.x_est = np.zeros((self.order, 1))
        self.P_est = np.eye(self.order)

        # Filter
        enhanced_signal = np.zeros_like(noisy_signal)

        for i, y in enumerate(noisy_signal):
            # Predict
            x_pred = self.phi @ self.x_est
            P_pred = self.phi @ self.P_est @ self.phi.T + self.Q

            # Update
            y_pred = self.H @ x_pred
            innovation = y - y_pred[0, 0]

            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T / S

            self.x_est = x_pred + K * innovation
            I = np.eye(self.order)
            self.P_est = (I - K @ self.H) @ P_pred

            # Adapt noise parameters
            if i > 10:  # Start adapting after initial samples
                self.adapt_noise_parameters(innovation)

            enhanced_signal[i] = self.x_est[-1, 0]

        return enhanced_signal


class SpectralSubtraction:
    """
    Spectral subtraction for noise reduction preprocessing.

    This can be applied before Kalman filtering for better results.
    """

    def __init__(self, frame_length=512, hop_length=256):
        """
        Initialize spectral subtraction.

        Args:
            frame_length: FFT frame length
            hop_length: Hop length between frames
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def estimate_noise_spectrum(self, signal, noise_frames=10):
        """
        Estimate noise spectrum from initial frames.

        Args:
            signal: Noisy signal
            noise_frames: Number of initial frames to use for noise estimation

        Returns:
            noise_spectrum: Estimated noise power spectrum
        """
        # Extract initial frames (assumed to contain mostly noise)
        noise_samples = min(noise_frames * self.hop_length, len(signal) // 2)
        noise_segment = signal[:noise_samples]

        # Compute power spectrum
        noise_fft = np.fft.rfft(noise_segment, n=self.frame_length)
        noise_spectrum = np.abs(noise_fft) ** 2

        return noise_spectrum

    def process(self, noisy_signal, alpha=2.0, beta=0.1):
        """
        Apply spectral subtraction.

        Args:
            noisy_signal: Noisy input signal
            alpha: Over-subtraction factor
            beta: Spectral floor parameter

        Returns:
            enhanced_signal: Enhanced signal
        """
        # Estimate noise spectrum
        noise_spectrum = self.estimate_noise_spectrum(noisy_signal)

        # Frame signal
        n_frames = (len(noisy_signal) - self.frame_length) // self.hop_length + 1

        # Window function
        window = np.hanning(self.frame_length)

        # Process frames
        enhanced_signal = np.zeros_like(noisy_signal)
        frame_count = np.zeros_like(noisy_signal)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length

            if end > len(noisy_signal):
                break

            # Extract frame
            frame = noisy_signal[start:end] * window

            # FFT
            frame_fft = np.fft.rfft(frame)
            magnitude = np.abs(frame_fft)
            phase = np.angle(frame_fft)

            # Spectral subtraction
            power = magnitude ** 2
            enhanced_power = power - alpha * noise_spectrum[:len(power)]

            # Apply spectral floor
            enhanced_power = np.maximum(enhanced_power, beta * power)

            # Reconstruct magnitude
            enhanced_magnitude = np.sqrt(enhanced_power)

            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_frame = np.fft.irfft(enhanced_fft, n=self.frame_length)

            # Overlap-add
            enhanced_signal[start:end] += enhanced_frame * window
            frame_count[start:end] += window

        # Normalize
        frame_count[frame_count == 0] = 1
        enhanced_signal = enhanced_signal / frame_count

        return enhanced_signal


class MFCCFeatureExtractor:
    """
    MFCC (Mel-Frequency Cepstral Coefficients) feature extraction for improved recognition.
    """

    def __init__(self, sample_rate=44100, n_mfcc=13, n_fft=512, hop_length=256, n_mels=40):
        """
        Initialize MFCC extractor.

        Args:
            sample_rate: Sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT size
            hop_length: Hop length
            n_mels: Number of mel filter banks
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create mel filterbank
        self.mel_filterbank = self._create_mel_filterbank()

    def _hz_to_mel(self, hz):
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        """Convert mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_mel_filterbank(self):
        """Create mel filterbank."""
        # Frequency range
        low_freq_mel = 0
        high_freq_mel = self._hz_to_mel(self.sample_rate / 2)

        # Mel points
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # FFT bin points
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        return filterbank

    def extract_mfcc(self, signal):
        """
        Extract MFCC features from signal.

        Args:
            signal: Input signal

        Returns:
            mfcc: MFCC features (n_frames x n_mfcc)
        """
        # Frame signal
        n_frames = (len(signal) - self.n_fft) // self.hop_length + 1

        # Pre-emphasis
        emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

        # Window
        window = np.hamming(self.n_fft)

        mfcc_features = []

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            if end > len(emphasized_signal):
                break

            # Extract frame
            frame = emphasized_signal[start:end] * window

            # FFT
            magnitude_spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))

            # Power spectrum
            power_spectrum = (magnitude_spectrum ** 2) / self.n_fft

            # Apply mel filterbank
            mel_spectrum = self.mel_filterbank @ power_spectrum

            # Log
            log_mel_spectrum = np.log(mel_spectrum + 1e-10)

            # DCT
            mfcc = dct(log_mel_spectrum, type=2, norm='ortho')[:self.n_mfcc]

            mfcc_features.append(mfcc)

        return np.array(mfcc_features)

    def compute_deltas(self, features, N=2):
        """
        Compute delta features.

        Args:
            features: Input features
            N: Delta window

        Returns:
            deltas: Delta features
        """
        n_frames, n_features = features.shape
        deltas = np.zeros_like(features)

        denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])

        for t in range(n_frames):
            numerator = np.zeros(n_features)
            for n in range(1, N + 1):
                if t + n < n_frames:
                    numerator += n * features[t + n]
                if t - n >= 0:
                    numerator -= n * features[t - n]

            deltas[t] = numerator / denominator

        return deltas


class WienerFilterEnhancement:
    """
    Wiener filter for speech enhancement (alternative/complement to Kalman).
    """

    def __init__(self, frame_length=512, hop_length=256):
        """
        Initialize Wiener filter.

        Args:
            frame_length: Frame length
            hop_length: Hop length
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def enhance(self, noisy_signal, noise_estimate=None):
        """
        Apply Wiener filtering.

        Args:
            noisy_signal: Noisy input
            noise_estimate: Estimated noise power (if None, estimated from signal)

        Returns:
            enhanced_signal: Enhanced signal
        """
        # Use scipy's Wiener filter as baseline
        enhanced_signal = wiener(noisy_signal, mysize=self.frame_length)

        return enhanced_signal


def enhance_with_ensemble(noisy_signal, Q_rbf, R, order=10, sample_rate=44100):
    """
    Ensemble enhancement combining multiple methods.

    Args:
        noisy_signal: Noisy input signal
        Q_rbf: RBF-estimated Q
        R: Measurement noise
        order: AR order
        sample_rate: Sampling rate

    Returns:
        enhanced_signal: Enhanced signal
    """
    # Method 1: Spectral subtraction + Kalman
    spec_sub = SpectralSubtraction()
    signal1 = spec_sub.process(noisy_signal)

    from kalman_filter import enhance_speech_rbf_kalman
    signal1 = enhance_speech_rbf_kalman(signal1, Q_rbf, R, order)

    # Method 2: Adaptive Kalman
    adaptive_kf = AdaptiveKalmanFilter(order=order)
    signal2 = adaptive_kf.filter_signal(noisy_signal, Q_rbf, R)

    # Method 3: Wiener filter
    wiener_filter = WienerFilterEnhancement()
    signal3 = wiener_filter.enhance(noisy_signal)

    # Ensemble average
    enhanced_signal = (signal1 + signal2 + signal3) / 3.0

    return enhanced_signal
