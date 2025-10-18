"""
Multi-Layer Radial Basis Function (RBF) Neural Network for Classification.

Implements a two-hidden-layer RBF network with Gaussian activations:
    Input → RBF Layer 1 (k1 centers) → RBF Layer 2 (k2 centers) → Softmax Output

Used for:
1. Speech segment classification (voiced/unvoiced/silence)
2. Word identification (hello, estimation, oakland, etc.)

Based on ChatGPT requirements for extended RBF network beyond single-layer.
"""

import numpy as np
from sklearn.cluster import KMeans


class MultiLayerRBFClassifier:
    """
    Two-layer RBF neural network for classification.

    Architecture:
        Input (d features)
          ↓
        RBF Layer 1 (k1 Gaussian centers, γ1)
          ↓ (k1 outputs)
        RBF Layer 2 (k2 Gaussian centers, γ2)
          ↓ (k2 outputs)
        Linear → Softmax (n_classes)
    """

    def __init__(self, k1=40, k2=20, gamma1=0.01, gamma2=0.05, n_classes=4, random_state=42):
        """
        Initialize multi-layer RBF classifier.

        Args:
            k1: Number of centers in first RBF layer
            k2: Number of centers in second RBF layer
            gamma1: Shape parameter for first layer
            gamma2: Shape parameter for second layer
            n_classes: Number of output classes
            random_state: Random seed for reproducibility
        """
        self.k1 = k1
        self.k2 = k2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.n_classes = n_classes
        self.random_state = random_state

        # Model parameters (set during training)
        self.centers1 = None  # First layer centers [k1, d]
        self.centers2 = None  # Second layer centers [k2, k1]
        self.W_out = None     # Output weights [k2, n_classes]
        self.b_out = None     # Output bias [n_classes]

    def _gaussian_rbf(self, X, centers, gamma):
        """
        Compute Gaussian RBF activations.

        Args:
            X: Input data [N, d]
            centers: RBF centers [k, d]
            gamma: Shape parameter

        Returns:
            activations: [N, k] RBF outputs
        """
        # Squared Euclidean distance
        # D²[i,j] = ||X[i] - centers[j]||²
        X = np.atleast_2d(X)
        centers = np.atleast_2d(centers)

        D_squared = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

        # Gaussian kernel
        activations = np.exp(-gamma * D_squared)

        return activations

    def _softmax(self, logits):
        """
        Compute softmax probabilities.

        Args:
            logits: [N, n_classes]

        Returns:
            probs: [N, n_classes] probabilities summing to 1
        """
        # Subtract max for numerical stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def fit(self, X, y, max_iter=100, learning_rate=0.01, verbose=True):
        """
        Train the multi-layer RBF network using gradient descent.

        Args:
            X: Training features [N, d]
            y: Training labels [N] (class indices 0..n_classes-1)
            max_iter: Maximum training iterations
            learning_rate: Learning rate for gradient descent
            verbose: Print training progress

        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        N, d = X.shape

        # Initialize first layer centers using k-means
        if verbose:
            print(f"Initializing Layer 1 centers (k1={self.k1})...")

        kmeans1 = KMeans(n_clusters=self.k1, random_state=self.random_state, n_init=10)
        kmeans1.fit(X)
        self.centers1 = kmeans1.cluster_centers_

        # Forward pass through layer 1 to get activations
        h1 = self._gaussian_rbf(X, self.centers1, self.gamma1)  # [N, k1]

        # Initialize second layer centers using k-means on h1
        if verbose:
            print(f"Initializing Layer 2 centers (k2={self.k2})...")

        kmeans2 = KMeans(n_clusters=self.k2, random_state=self.random_state, n_init=10)
        kmeans2.fit(h1)
        self.centers2 = kmeans2.cluster_centers_

        # Initialize output layer weights randomly
        self.W_out = np.random.randn(self.k2, self.n_classes) * 0.01
        self.b_out = np.zeros(self.n_classes)

        # Convert labels to one-hot encoding
        y_onehot = np.zeros((N, self.n_classes))
        y_onehot[np.arange(N), y] = 1

        # Training loop (fine-tune output weights with fixed RBF centers)
        if verbose:
            print(f"Training output layer...")

        for epoch in range(max_iter):
            # Forward pass
            h1 = self._gaussian_rbf(X, self.centers1, self.gamma1)
            h2 = self._gaussian_rbf(h1, self.centers2, self.gamma2)

            # Logits and probabilities
            logits = h2 @ self.W_out + self.b_out
            probs = self._softmax(logits)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))

            # Gradient of cross-entropy w.r.t. W_out and b_out
            error = probs - y_onehot  # [N, n_classes]

            grad_W = h2.T @ error / N
            grad_b = np.mean(error, axis=0)

            # Update weights
            self.W_out -= learning_rate * grad_W
            self.b_out -= learning_rate * grad_b

            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"  Epoch {epoch+1}/{max_iter} - Loss: {loss:.4f}, Acc: {acc:.4f}")

        if verbose:
            final_acc = np.mean(self.predict(X) == y)
            print(f"✓ Training complete. Final accuracy: {final_acc:.4f}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Input features [N, d]

        Returns:
            probs: Class probabilities [N, n_classes]
        """
        if self.centers1 is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.atleast_2d(X)

        # Forward pass
        h1 = self._gaussian_rbf(X, self.centers1, self.gamma1)
        h2 = self._gaussian_rbf(h1, self.centers2, self.gamma2)
        logits = h2 @ self.W_out + self.b_out
        probs = self._softmax(logits)

        return probs

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Input features [N, d]

        Returns:
            labels: Predicted class indices [N]
        """
        probs = self.predict_proba(X)
        labels = np.argmax(probs, axis=1)

        return labels

    def score(self, X, y):
        """
        Compute accuracy.

        Args:
            X: Features
            y: True labels

        Returns:
            accuracy: Fraction of correct predictions
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing MultiLayerRBFClassifier...")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data (4 classes, 10 features)
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=4,
        random_state=42
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Create and train model
    model = MultiLayerRBFClassifier(
        k1=30,
        k2=15,
        gamma1=0.05,
        gamma2=0.1,
        n_classes=4,
        random_state=42
    )

    model.fit(X_train, y_train, max_iter=100, learning_rate=0.05, verbose=True)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Test prediction
    y_pred = model.predict(X_test[:5])
    y_proba = model.predict_proba(X_test[:5])

    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  True: {y_test[i]}, Pred: {y_pred[i]}, Probs: {y_proba[i]}")

    print("\n✓ MultiLayerRBFClassifier test complete")
