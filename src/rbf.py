"""
Radial Basis Function (RBF) implementation for Kalman filter modification.
Based on the paper: Speech Enhancement and Recognition using Kalman Filter Modified via RBF
"""

import numpy as np
from scipy.spatial.distance import cdist


class RadialBasisFunction:
    """
    Radial Basis Function Network for estimating process noise covariance Q.

    The RBF is used to provide non-linear estimation of the Q parameter in the Kalman filter,
    allowing for better noise adaptation compared to linear methods.
    """

    def __init__(self, gamma=1.0):
        """
        Initialize RBF.

        Args:
            gamma: Shape parameter for RBF (affects the width of the Gaussian basis functions)
        """
        self.gamma = gamma
        self.weights = None
        self.centers = None

    def gaussian_rbf(self, x, centers):
        """
        Compute Gaussian RBF kernel.

        Args:
            x: Input data points (N x D)
            centers: RBF centers (M x D)

        Returns:
            Phi matrix (N x M) containing RBF activations
        """
        # Compute Euclidean distances between x and centers
        distances_squared = cdist(x, centers, metric='sqeuclidean')

        # Apply Gaussian kernel: exp(-gamma * ||x - c||^2)
        phi = np.exp(-self.gamma * distances_squared)

        return phi

    def fit(self, X, y):
        """
        Train the RBF network.

        Args:
            X: Training data (N x D) - centers of dataset
            y: Target values (N x 1)

        According to paper equation (2):
            Phi * W = Y
            W = Phi^(-1) * Y
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Use training data points as centers
        self.centers = X.copy()

        # Compute Phi matrix
        phi = self.gaussian_rbf(X, self.centers)

        # Add small regularization to avoid singularity
        regularization = 1e-8 * np.eye(phi.shape[1])

        # Solve for weights: W = (Phi^T * Phi + lambda*I)^(-1) * Phi^T * y
        try:
            self.weights = np.linalg.solve(phi.T @ phi + regularization, phi.T @ y)
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            self.weights = np.linalg.pinv(phi) @ y

        return self

    def predict(self, X):
        """
        Predict using trained RBF network.

        Args:
            X: Input data (N x D)

        Returns:
            Predictions (N,)
        """
        if self.weights is None:
            raise ValueError("RBF network must be trained before prediction")

        X = np.atleast_2d(X)

        # Compute Phi matrix for new data
        phi = self.gaussian_rbf(X, self.centers)

        # Compute predictions: y = Phi * W
        predictions = phi @ self.weights

        return predictions

    def estimate_process_noise(self, signal_variance, num_samples=100):
        """
        Estimate process noise covariance Q using RBF.

        This provides non-linear estimation of Q based on signal characteristics,
        which is the key innovation in the paper.

        Args:
            signal_variance: Variance of the noisy signal
            num_samples: Number of samples to use for estimation

        Returns:
            Estimated Q values (array)
        """
        # Create synthetic training data based on signal variance
        # X: variance values, y: corresponding Q values
        X_train = np.linspace(0.01, signal_variance * 2, num_samples).reshape(-1, 1)

        # Non-linear relationship between variance and Q
        # This is a heuristic based on typical speech signal characteristics
        y_train = 0.5 * X_train.flatten() * (1 + 0.3 * np.sin(2 * np.pi * X_train.flatten() / signal_variance))

        # Train RBF
        self.fit(X_train, y_train)

        # Predict Q for current signal variance
        Q_estimate = self.predict(np.array([[signal_variance]]))

        return max(Q_estimate[0], 1e-6)  # Ensure positive Q


def create_rbf_for_kalman(signal, gamma=1.0):
    """
    Create and configure RBF for Kalman filter Q estimation.

    Args:
        signal: Input noisy signal
        gamma: RBF shape parameter

    Returns:
        Configured RBF instance with estimated Q
    """
    rbf = RadialBasisFunction(gamma=gamma)

    # Compute signal variance
    signal_variance = np.var(signal)

    # Estimate Q using RBF
    Q = rbf.estimate_process_noise(signal_variance)

    return rbf, Q
