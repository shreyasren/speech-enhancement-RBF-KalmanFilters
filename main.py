"""
Main experiment pipeline for Speech Enhancement and Recognition using RBF-Kalman Filter.

This script recreates the experiments from the paper:
"Speech Enhancement and Recognition using Kalman Filter Modified via Radial Basis Function"
by Mario Barnard, Farag M. Lagnf, Amr S. Mahmoud, Mohamed Zohdy (2020)
"""

import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_utils import AudioRecorder, AudioProcessor, create_dataset_structure
from rbf import RadialBasisFunction, create_rbf_for_kalman
from kalman_filter import KalmanFilterSpeech, enhance_speech_rbf_kalman
from envelope_detection import EnvelopeDetector, detect_envelope, remove_silence_from_signal
from speech_recognition import CorrelationSpeechRecognizer, DTWSpeechRecognizer
from visualization import SpeechVisualizer


class SpeechEnhancementExperiment:
    """
    Complete experiment pipeline for speech enhancement and recognition.
    """

    def __init__(self, sample_rate=44100, duration=3.0, ar_order=10):
        """
        Initialize experiment.

        Args:
            sample_rate: Audio sampling rate (Hz)
            duration: Recording duration (seconds)
            ar_order: Autoregressive model order
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.ar_order = ar_order

        # Initialize components
        self.recorder = AudioRecorder(sample_rate, duration)
        self.processor = AudioProcessor(sample_rate)
        self.envelope_detector = EnvelopeDetector()
        self.recognizer = CorrelationSpeechRecognizer()
        self.dtw_recognizer = DTWSpeechRecognizer()
        self.visualizer = SpeechVisualizer()

        # Create directory structure
        self.paths = create_dataset_structure()

        # Database
        self.database = {}
        self.words = ['Hello', 'Estimation', 'Oakland']

    def record_database(self, num_people=3):
        """
        Record speech database from multiple people.

        Args:
            num_people: Number of different speakers
        """
        print("=" * 60)
        print("RECORDING DATABASE")
        print("=" * 60)

        for person_id in range(1, num_people + 1):
            print(f"\n--- Person {person_id} ---")
            input(f"Press Enter when Person {person_id} is ready to record...")

            for word in self.words:
                key = f"{word}_person{person_id}"
                message = f"Say '{word}'"

                # Record
                audio = self.recorder.record(message)

                # Normalize
                audio = self.processor.normalize(audio)

                # Save raw recording
                filename = os.path.join(self.paths['raw'], f"{key}.wav")
                self.recorder.save(filename, audio)

                # Store in database
                self.database[key] = audio

        print(f"\nRecorded {len(self.database)} samples")

    def load_database_from_files(self):
        """
        Load database from saved WAV files.
        """
        print("Loading database from files...")

        raw_path = self.paths['raw']
        files = [f for f in os.listdir(raw_path) if f.endswith('.wav')]

        for file in files:
            filepath = os.path.join(raw_path, file)
            audio, sr = self.processor.load_audio(filepath)

            if audio is not None:
                # Resample if necessary
                if sr != self.sample_rate:
                    audio = self.processor.resample(audio, sr, self.sample_rate)

                # Normalize
                audio = self.processor.normalize(audio)

                # Store in database
                key = file.replace('.wav', '')
                self.database[key] = audio

        print(f"Loaded {len(self.database)} samples from files")

    def enhance_database(self, snr_db=10, rbf_gamma=1.0, measurement_noise=0.01):
        """
        Enhance all database signals using RBF-Kalman filter.

        Args:
            snr_db: Signal-to-Noise Ratio for added noise (dB)
            rbf_gamma: RBF shape parameter
            measurement_noise: Measurement noise covariance R
        """
        print("\n" + "=" * 60)
        print("ENHANCING DATABASE SIGNALS")
        print("=" * 60)

        enhanced_database = {}

        for key, clean_signal in self.database.items():
            print(f"\nProcessing: {key}")

            # Add noise
            noisy_signal, noise = self.processor.add_noise(clean_signal, snr_db)

            # Estimate Q using RBF
            rbf, Q = create_rbf_for_kalman(noisy_signal, gamma=rbf_gamma)
            print(f"  Estimated Q (process noise): {Q:.6f}")

            # Apply Kalman filter
            enhanced_signal = enhance_speech_rbf_kalman(
                noisy_signal,
                Q_rbf=Q,
                R=measurement_noise,
                order=self.ar_order
            )

            # Normalize enhanced signal
            enhanced_signal = self.processor.normalize(enhanced_signal)

            # Save enhanced signal
            filename = os.path.join(self.paths['database'], f"{key}_enhanced.wav")
            self.recorder.save(filename, enhanced_signal)

            enhanced_database[key] = {
                'clean': clean_signal,
                'noisy': noisy_signal,
                'enhanced': enhanced_signal,
                'noise': noise,
                'Q': Q
            }

        print(f"\nEnhanced {len(enhanced_database)} signals")
        return enhanced_database

    def apply_envelope_detection(self, enhanced_database):
        """
        Apply envelope detection to remove silent portions.

        Args:
            enhanced_database: Dictionary of enhanced signals
        """
        print("\n" + "=" * 60)
        print("APPLYING ENVELOPE DETECTION")
        print("=" * 60)

        voiced_database = {}

        for key, data in enhanced_database.items():
            print(f"\nProcessing: {key}")

            signal = data['enhanced']

            # Detect envelope
            envelope, voiced_mask = detect_envelope(signal)

            # Extract voiced signal
            voiced_signal = remove_silence_from_signal(signal)

            print(f"  Original samples: {len(signal)}")
            print(f"  Voiced samples: {len(voiced_signal)}")
            print(f"  Reduction: {(1 - len(voiced_signal) / len(signal)) * 100:.1f}%")

            voiced_database[key] = {
                'signal': signal,
                'envelope': envelope,
                'voiced_mask': voiced_mask,
                'voiced_signal': voiced_signal
            }

            # Save voiced signal
            filename = os.path.join(self.paths['processed'], f"{key}_voiced.wav")
            self.recorder.save(filename, voiced_signal)

        return voiced_database

    def build_recognition_database(self, voiced_database):
        """
        Build database for speech recognition.

        Args:
            voiced_database: Dictionary of voiced signals
        """
        print("\n" + "=" * 60)
        print("BUILDING RECOGNITION DATABASE")
        print("=" * 60)

        for key, data in voiced_database.items():
            # Extract word label
            word = key.split('_')[0]

            # Add to recognizers
            voiced_signal = data['voiced_signal']
            self.recognizer.add_to_database(word, voiced_signal)
            self.dtw_recognizer.add_to_database(word, voiced_signal)

        print(f"Database contains {len(self.recognizer.labels)} words")
        for word in self.recognizer.labels:
            count = len(self.recognizer.database[word])
            print(f"  {word}: {count} samples")

    def test_recognition(self, test_word, snr_db=10):
        """
        Test speech recognition with a new recording.

        Args:
            test_word: Word to test
            snr_db: SNR for added noise
        """
        print("\n" + "=" * 60)
        print(f"TESTING RECOGNITION: {test_word}")
        print("=" * 60)

        # Record test signal
        print(f"\nRecording test signal for '{test_word}'")
        test_signal = self.recorder.record(f"Say '{test_word}'")

        # Normalize
        test_signal = self.processor.normalize(test_signal)

        # Add noise
        noisy_test, _ = self.processor.add_noise(test_signal, snr_db)

        # Enhance using RBF-Kalman
        rbf, Q = create_rbf_for_kalman(noisy_test)
        enhanced_test = enhance_speech_rbf_kalman(noisy_test, Q_rbf=Q, R=0.01, order=self.ar_order)

        # Remove silence
        voiced_test = remove_silence_from_signal(enhanced_test)

        # Recognize using correlation
        recognized_word, confidence, all_scores = self.recognizer.recognize(voiced_test)

        print(f"\nCorrelation-based Recognition:")
        print(f"  Recognized: {recognized_word}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  All scores: {all_scores}")

        # Recognize using DTW
        dtw_word, dtw_distance, all_distances = self.dtw_recognizer.recognize(voiced_test)

        print(f"\nDTW-based Recognition:")
        print(f"  Recognized: {dtw_word}")
        print(f"  Distance: {dtw_distance:.3f}")
        print(f"  All distances: {all_distances}")

        # Save test signal
        test_key = f"test_{test_word}"
        filename = os.path.join(self.paths['test'], f"{test_key}.wav")
        self.recorder.save(filename, enhanced_test)

        return {
            'test_signal': test_signal,
            'noisy': noisy_test,
            'enhanced': enhanced_test,
            'voiced': voiced_test,
            'recognized_corr': recognized_word,
            'confidence_corr': confidence,
            'scores_corr': all_scores,
            'recognized_dtw': dtw_word,
            'distance_dtw': dtw_distance,
            'distances_dtw': all_distances
        }

    def visualize_results(self, enhanced_database, voiced_database):
        """
        Generate visualizations matching paper figures.

        Args:
            enhanced_database: Enhanced signals
            voiced_database: Voiced signals
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        results_path = self.paths['results']

        # Figure 1: Time domain of "Estimation" from 3 people
        estimation_signals = []
        estimation_titles = []
        for i in range(1, 4):
            key = f"Estimation_person{i}"
            if key in self.database:
                estimation_signals.append(self.database[key])
                estimation_titles.append(f'{key.split("_")[1].capitalize()} Says "Estimation"')

        if estimation_signals:
            self.visualizer.plot_time_domain(
                estimation_signals,
                estimation_titles,
                self.sample_rate,
                save_path=os.path.join(results_path, 'figure1_time_domain_estimation.png')
            )

        # Figure 2: Frequency domain of "Estimation" from 3 people
        if estimation_signals:
            self.visualizer.plot_frequency_domain(
                estimation_signals,
                estimation_titles,
                self.sample_rate,
                save_path=os.path.join(results_path, 'figure2_frequency_domain_estimation.png')
            )

        # Figure 7: Enhancement results
        for key in list(enhanced_database.keys())[:3]:  # Show first 3 examples
            data = enhanced_database[key]
            self.visualizer.plot_enhancement_results(
                data['clean'],
                data['noisy'],
                data['enhanced'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure7_enhancement_{key}.png')
            )

        # Envelope detection visualization
        for key in list(voiced_database.keys())[:3]:
            data = voiced_database[key]
            self.visualizer.plot_envelope_detection(
                data['signal'],
                data['envelope'],
                data['voiced_mask'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure5_envelope_{key}.png')
            )

            self.visualizer.plot_voiced_signal(
                data['voiced_signal'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure6_voiced_{key}.png')
            )

        print(f"\nVisualizations saved to: {results_path}")


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("SPEECH ENHANCEMENT AND RECOGNITION USING RBF-KALMAN FILTER")
    print("=" * 60)

    # Initialize experiment
    exp = SpeechEnhancementExperiment(
        sample_rate=44100,
        duration=3.0,
        ar_order=10
    )

    # Choose mode
    print("\nChoose mode:")
    print("1. Record new database")
    print("2. Load existing database")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        # Record database
        exp.record_database(num_people=3)
    else:
        # Load existing database
        exp.load_database_from_files()

    if not exp.database:
        print("No database available. Exiting.")
        return

    # Enhance database
    enhanced_db = exp.enhance_database(snr_db=10, rbf_gamma=1.0, measurement_noise=0.01)

    # Apply envelope detection
    voiced_db = exp.apply_envelope_detection(enhanced_db)

    # Build recognition database
    exp.build_recognition_database(voiced_db)

    # Visualize results
    exp.visualize_results(enhanced_db, voiced_db)

    # Test recognition
    print("\n" + "=" * 60)
    print("SPEECH RECOGNITION TEST")
    print("=" * 60)

    test_mode = input("\nTest recognition? (y/n): ").strip().lower()

    if test_mode == 'y':
        for word in exp.words:
            test_again = input(f"\nTest word '{word}'? (y/n): ").strip().lower()
            if test_again == 'y':
                result = exp.test_recognition(word, snr_db=10)

                # Visualize recognition result
                exp.visualizer.plot_recognition_results(
                    result['voiced'],
                    result['recognized_corr'],
                    result['confidence_corr'],
                    result['scores_corr'],
                    exp.sample_rate,
                    save_path=os.path.join(exp.paths['results'], f'recognition_test_{word}.png')
                )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved in: {exp.paths['results']}")


if __name__ == "__main__":
    main()
