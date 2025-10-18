"""
Speech recognition using correlation matching.
Based on the paper: Speech Enhancement and Recognition using Kalman Filter Modified via RBF
"""

import numpy as np
from scipy import signal as sp_signal


class CorrelationSpeechRecognizer:
    """
    Speech recognition using correlation-based template matching.

    The paper states: "speech recognition would be accomplished by matching a test
    voice signal with the database" using correlation after envelope detection.
    """

    def __init__(self):
        """
        Initialize speech recognizer.
        """
        self.database = {}
        self.labels = []

    def add_to_database(self, label, signal):
        """
        Add a speech sample to the database.

        Args:
            label: Label/word for the speech sample
            signal: Enhanced speech signal
        """
        if label not in self.database:
            self.database[label] = []
            self.labels.append(label)

        self.database[label].append(signal)

    def compute_correlation(self, signal1, signal2):
        """
        Compute normalized cross-correlation between two signals.

        Args:
            signal1: First signal
            signal2: Second signal

        Returns:
            max_correlation: Maximum correlation value
            lag: Lag at maximum correlation
        """
        # Ensure signals are 1D
        signal1 = np.asarray(signal1).flatten()
        signal2 = np.asarray(signal2).flatten()

        # Normalize signals
        signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

        # Compute cross-correlation
        correlation = np.correlate(signal1, signal2, mode='full')

        # Normalize by length
        n = min(len(signal1), len(signal2))
        correlation = correlation / n

        # Find maximum correlation
        max_correlation = np.max(np.abs(correlation))
        lag = np.argmax(np.abs(correlation)) - len(signal2) + 1

        return max_correlation, lag

    def compute_autocorrelation(self, signal):
        """
        Compute autocorrelation of a signal.

        The paper mentions: "Those phrases will be identified if the sound waves
        are primarily auto-correlated. It would double the data amplitude."

        Args:
            signal: Input signal

        Returns:
            autocorr: Autocorrelation values
        """
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')

        # Normalize
        autocorr = autocorr / np.max(autocorr)

        return autocorr

    def match_signal(self, test_signal, method='correlation'):
        """
        Match test signal against database using correlation.

        Args:
            test_signal: Test speech signal
            method: Matching method ('correlation' or 'autocorrelation')

        Returns:
            best_match: Label of best matching word
            best_score: Correlation score
            all_scores: Dictionary of scores for all labels
        """
        all_scores = {}

        for label in self.labels:
            max_score = 0

            # Compare with all samples of this label
            for reference_signal in self.database[label]:
                if method == 'correlation':
                    score, _ = self.compute_correlation(test_signal, reference_signal)
                elif method == 'autocorrelation':
                    # Use autocorrelation difference as similarity metric
                    test_autocorr = self.compute_autocorrelation(test_signal)
                    ref_autocorr = self.compute_autocorrelation(reference_signal)

                    # Compute similarity based on autocorrelation patterns
                    min_len = min(len(test_autocorr), len(ref_autocorr))
                    diff = np.abs(test_autocorr[:min_len] - ref_autocorr[:min_len])
                    score = 1.0 - np.mean(diff)  # Convert difference to similarity
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Keep maximum score
                max_score = max(max_score, score)

            all_scores[label] = max_score

        # Find best match
        best_match = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_match]

        return best_match, best_score, all_scores

    def recognize(self, test_signal, threshold=0.5):
        """
        Recognize speech from test signal.

        Args:
            test_signal: Test speech signal
            threshold: Minimum correlation threshold for recognition

        Returns:
            recognized_word: Recognized word (or 'Unknown')
            confidence: Confidence score
            all_scores: Scores for all words
        """
        best_match, best_score, all_scores = self.match_signal(test_signal)

        if best_score >= threshold:
            recognized_word = best_match
            confidence = best_score
        else:
            recognized_word = 'Unknown'
            confidence = 0.0

        return recognized_word, confidence, all_scores


class DTWSpeechRecognizer:
    """
    Dynamic Time Warping based speech recognition (enhanced method).

    This is an improvement over simple correlation matching.
    """

    def __init__(self):
        """
        Initialize DTW-based recognizer.
        """
        self.database = {}
        self.labels = []

    def add_to_database(self, label, signal):
        """
        Add a speech sample to the database.

        Args:
            label: Label/word for the speech sample
            signal: Enhanced speech signal
        """
        if label not in self.database:
            self.database[label] = []
            self.labels.append(label)

        self.database[label].append(signal)

    def dtw_distance(self, signal1, signal2):
        """
        Compute Dynamic Time Warping distance between two signals.

        Args:
            signal1: First signal
            signal2: Second signal

        Returns:
            distance: DTW distance
        """
        n, m = len(signal1), len(signal2)

        # Initialize DTW matrix
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[0, :] = np.inf
        dtw_matrix[:, 0] = np.inf
        dtw_matrix[0, 0] = 0

        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (signal1[i - 1] - signal2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1]   # match
                )

        # Return normalized distance
        distance = np.sqrt(dtw_matrix[n, m]) / (n + m)

        return distance

    def recognize(self, test_signal):
        """
        Recognize speech using DTW matching.

        Args:
            test_signal: Test speech signal

        Returns:
            recognized_word: Recognized word
            distance: DTW distance to best match
            all_distances: Distances to all words
        """
        # Normalize test signal
        test_signal = (test_signal - np.mean(test_signal)) / (np.std(test_signal) + 1e-10)

        all_distances = {}

        for label in self.labels:
            min_distance = np.inf

            # Compare with all samples of this label
            for reference_signal in self.database[label]:
                # Normalize reference signal
                ref_signal = (reference_signal - np.mean(reference_signal)) / (np.std(reference_signal) + 1e-10)

                # Compute DTW distance
                distance = self.dtw_distance(test_signal, ref_signal)

                # Keep minimum distance
                min_distance = min(min_distance, distance)

            all_distances[label] = min_distance

        # Find best match (minimum distance)
        recognized_word = min(all_distances, key=all_distances.get)
        distance = all_distances[recognized_word]

        return recognized_word, distance, all_distances
