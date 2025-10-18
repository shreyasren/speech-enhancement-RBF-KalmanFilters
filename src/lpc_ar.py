"""
Proper LPC (Linear Predictive Coding) to AR State-Space Conversion.

Implements autocorrelation method for LPC coefficient estimation and
builds companion-form state-space matrices matching paper Equations (3-5).

Based on ChatGPT requirements for faithful paper reproduction.
"""

import numpy as np
from scipy.linalg import toeplitz, solve


def lpc_autocorrelation(signal, order=12):
    """
    Compute LPC coefficients using autocorrelation method.

    This is the standard method: compute autocorrelation r[0..p],
    then solve the Toeplitz system to find prediction coefficients.

    Args:
        signal: Input signal
        order: LPC order (p)

    Returns:
        a: LPC polynomial coefficients [1, -a1, -a2, ..., -ap]
            where A(z) = 1 + sum(a_k * z^-k) for k=1..p
    """
    # Ensure 1D
    signal = np.asarray(signal).flatten()

    # Compute autocorrelation using numpy correlate
    autocorr = np.correlate(signal, signal, mode='full')

    # Take the second half (lag >= 0) and normalize
    autocorr = autocorr[len(signal)-1:]

    # We need r[0] through r[order]
    r = autocorr[:order+1]

    # Normalize by r[0] to avoid numerical issues
    r = r / (r[0] + 1e-10)

    # Build Toeplitz matrix R from r[0..p-1]
    # The Yule-Walker equations: R * a = r[1..p]
    R = toeplitz(r[:-1])

    # Right-hand side
    rhs = r[1:]

    try:
        # Solve for prediction coefficients
        a = solve(R, rhs, assume_a='pos')
    except np.linalg.LinAlgError:
        # If singular, use least squares
        a = np.linalg.lstsq(R, rhs, rcond=None)[0]

    # Return as [1, -a1, -a2, ..., -ap] (standard LPC polynomial form)
    return np.r_[1.0, -a]


def companion_from_lpc(lpc_coeffs):
    """
    Build companion-form state-space matrices from LPC coefficients.

    Matches paper Equations (3-5):
        x(k) = Φ·x(k-1) + G·u(k)
        y(k) = H·x(k) + w(k)

    Args:
        lpc_coeffs: LPC polynomial [1, -a1, -a2, ..., -ap]

    Returns:
        Phi: State transition matrix (p × p)
        G: Input matrix (p × 1)
        H: Observation matrix (1 × p)
    """
    # Extract prediction coefficients (skip the leading 1)
    a = lpc_coeffs[1:]  # [-a1, -a2, ..., -ap]
    p = len(a)

    # Build companion matrix Phi
    # Structure:
    #   [ -a_p  -a_{p-1}  ...  -a_2  -a_1 ]
    #   [   1       0      ...    0     0  ]
    #   [   0       1      ...    0     0  ]
    #   ...
    #   [   0       0      ...    1     0  ]

    Phi = np.zeros((p, p))

    # Top row: -a_p, -a_{p-1}, ..., -a_1 (reversed)
    Phi[0, :] = -a[::-1]

    # Sub-diagonal: identity shift
    if p > 1:
        Phi[1:, :-1] = np.eye(p-1)

    # Input matrix G (excitation enters first state)
    G = np.zeros((p, 1))
    G[0, 0] = 1.0

    # Observation matrix H (observe last state)
    H = np.zeros((1, p))
    H[0, -1] = 1.0

    return Phi, G, H


def lpc_state_space(signal, order=12):
    """
    Convenience function: signal → LPC → state-space matrices.

    Args:
        signal: Input signal
        order: LPC order

    Returns:
        Phi: State transition (p × p)
        G: Input (p × 1)
        H: Observation (1 × p)
        a: LPC coefficients [1, -a1, ..., -ap]
    """
    lpc_coeffs = lpc_autocorrelation(signal, order)
    Phi, G, H = companion_from_lpc(lpc_coeffs)

    return Phi, G, H, lpc_coeffs


def estimate_excitation_variance(signal, lpc_coeffs):
    """
    Estimate variance of excitation (prediction error).

    Useful for setting initial Q in Kalman filter.

    Args:
        signal: Input signal
        lpc_coeffs: LPC polynomial [1, -a1, ..., -ap]

    Returns:
        sigma_e: Excitation variance
    """
    # Filter signal with inverse LPC: e = A(z) * s
    from scipy.signal import lfilter

    excitation = lfilter(lpc_coeffs, [1.0], signal)

    # Variance of excitation
    sigma_e = np.var(excitation)

    return sigma_e


if __name__ == "__main__":
    # Test with synthetic AR(2) signal
    print("Testing LPC-AR module...")

    # Generate AR(2): x[n] = 0.9*x[n-1] - 0.4*x[n-2] + e[n]
    np.random.seed(42)
    N = 1000
    x = np.zeros(N)
    e = np.random.randn(N) * 0.1

    for n in range(2, N):
        x[n] = 0.9*x[n-1] - 0.4*x[n-2] + e[n]

    # Estimate LPC
    lpc_coeffs = lpc_autocorrelation(x, order=2)
    print(f"True coefficients: [1, -0.9, 0.4]")
    print(f"Estimated LPC: {lpc_coeffs}")

    # Build state-space
    Phi, G, H = companion_from_lpc(lpc_coeffs)
    print(f"\nPhi (should have 0.9, -0.4 in top row):\n{Phi}")
    print(f"\nG:\n{G.T}")
    print(f"\nH:\n{H}")

    # Estimate excitation variance
    sigma_e = estimate_excitation_variance(x, lpc_coeffs)
    print(f"\nTrue excitation variance: {0.01:.6f}")
    print(f"Estimated variance: {sigma_e:.6f}")

    print("\n✓ LPC-AR module test complete")
