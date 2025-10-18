"""
Kalman Filter implementation for speech enhancement.
Based on the paper: Speech Enhancement and Recognition using Kalman Filter Modified via RBF
"""

import numpy as np
from scipy.signal import lfilter
from scipy.linalg import solve_discrete_are


class KalmanFilterSpeech:
    """
    Kalman Filter for speech enhancement using AR model.

    State space model:
        X(k) = φ*X(k-1) + G*u(k)    [State equation]
        y(k) = H*X(k) + w(k)         [Measurement equation]

    where:
        φ: State transition matrix (from AR coefficients)
        G: Input matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
    """

    def __init__(self, order=10):
        """
        Initialize Kalman filter.

        Args:
            order: AR model order (p in the paper)
        """
        self.order = order
        self.phi = None  # State transition matrix
        self.G = None    # Input matrix
        self.H = None    # Observation matrix
        self.Q = None    # Process noise covariance
        self.R = None    # Measurement noise covariance

        # Kalman filter state variables
        self.x_pred = None  # Predicted state
        self.x_est = None   # Estimated state
        self.P_pred = None  # Predicted error covariance
        self.P_est = None   # Estimated error covariance

    def estimate_ar_coefficients(self, signal):
        """
        Estimate AR coefficients using Levinson-Durbin algorithm.

        Args:
            signal: Input signal

        Returns:
            ar_coeffs: AR coefficients [a1, a2, ..., ap]
        """
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Normalize
        autocorr = autocorr / autocorr[0]

        # Levinson-Durbin recursion
        ar_coeffs = np.zeros(self.order)
        E = autocorr[0]

        for i in range(self.order):
            if i == 0:
                k = autocorr[1] / E
                ar_coeffs[0] = k
                E = E * (1 - k ** 2)
            else:
                # Compute reflection coefficient
                sum_val = autocorr[i + 1]
                for j in range(i):
                    sum_val -= ar_coeffs[j] * autocorr[i - j]

                k = sum_val / E

                # Update AR coefficients
                ar_coeffs_old = ar_coeffs.copy()
                ar_coeffs[i] = k
                for j in range(i):
                    ar_coeffs[j] = ar_coeffs_old[j] - k * ar_coeffs_old[i - 1 - j]

                E = E * (1 - k ** 2)

        return ar_coeffs

    def build_state_space_model(self, ar_coeffs):
        """
        Build state space matrices from AR coefficients.

        Args:
            ar_coeffs: AR coefficients [a1, a2, ..., ap]
        """
        p = len(ar_coeffs)

        # State transition matrix φ (p x p)
        # Companion form of AR model
        self.phi = np.zeros((p, p))
        self.phi[0, :] = ar_coeffs
        if p > 1:
            self.phi[1:, :-1] = np.eye(p - 1)

        # Input matrix G (p x 1)
        self.G = np.zeros((p, 1))
        self.G[0, 0] = 1.0

        # Observation matrix H (1 x p)
        self.H = np.zeros((1, p))
        self.H[0, -1] = 1.0

    def initialize_filter(self, noisy_signal, Q, R):
        """
        Initialize Kalman filter parameters.

        Args:
            noisy_signal: Noisy input signal
            Q: Process noise covariance (estimated by RBF)
            R: Measurement noise covariance
        """
        # Estimate AR coefficients from noisy signal
        ar_coeffs = self.estimate_ar_coefficients(noisy_signal)

        # Build state space model
        self.build_state_space_model(ar_coeffs)

        # Set noise covariances
        self.Q = Q * np.eye(self.order)  # Process noise
        self.R = R  # Measurement noise

        # Initialize state estimate
        self.x_est = np.zeros((self.order, 1))

        # Initialize error covariance
        # Use steady-state solution of discrete algebraic Riccati equation
        try:
            self.P_est = solve_discrete_are(self.phi.T, self.H.T, self.Q, self.R)
        except:
            self.P_est = np.eye(self.order)

    def predict(self, u):
        """
        Kalman filter prediction step.

        Args:
            u: Input (process noise)
        """
        # Predicted state: x_pred = φ * x_est + G * u
        self.x_pred = self.phi @ self.x_est + self.G * u

        # Predicted error covariance: P_pred = φ * P_est * φ^T + Q
        self.P_pred = self.phi @ self.P_est @ self.phi.T + self.Q

    def update(self, y):
        """
        Kalman filter update step.

        Args:
            y: Measurement (noisy observation)
        """
        # Innovation: y_tilde = y - H * x_pred
        y_pred = self.H @ self.x_pred
        innovation = y - y_pred

        # Innovation covariance: S = H * P_pred * H^T + R
        S = self.H @ self.P_pred @ self.H.T + self.R

        # Kalman gain: K = P_pred * H^T * S^(-1)
        K = self.P_pred @ self.H.T / S

        # Updated state estimate: x_est = x_pred + K * innovation
        self.x_est = self.x_pred + K * innovation

        # Updated error covariance: P_est = (I - K*H) * P_pred
        I = np.eye(self.order)
        self.P_est = (I - K @ self.H) @ self.P_pred

        # Extract enhanced signal (last state)
        enhanced_sample = self.x_est[-1, 0]

        return enhanced_sample

    def filter_signal(self, noisy_signal, Q, R):
        """
        Apply Kalman filter to entire signal.

        Args:
            noisy_signal: Input noisy signal
            Q: Process noise covariance (from RBF)
            R: Measurement noise covariance

        Returns:
            enhanced_signal: Filtered signal
        """
        # Initialize filter
        self.initialize_filter(noisy_signal, Q, R)

        # Process each sample
        enhanced_signal = np.zeros_like(noisy_signal)

        for i, y in enumerate(noisy_signal):
            # Prediction
            u = 0  # Assuming zero-mean process noise
            self.predict(u)

            # Update
            enhanced_sample = self.update(y)
            enhanced_signal[i] = enhanced_sample

        return enhanced_signal


def enhance_speech_rbf_kalman(noisy_signal, Q_rbf, R=0.01, order=10):
    """
    Enhance speech using RBF-modified Kalman filter.

    Args:
        noisy_signal: Noisy input speech signal
        Q_rbf: Process noise covariance estimated by RBF
        R: Measurement noise covariance
        order: AR model order

    Returns:
        enhanced_signal: Enhanced speech signal
    """
    kf = KalmanFilterSpeech(order=order)
    enhanced_signal = kf.filter_signal(noisy_signal, Q_rbf, R)

    return enhanced_signal
