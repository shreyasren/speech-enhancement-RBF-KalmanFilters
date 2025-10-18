"""
Audio utilities for recording, loading, and preprocessing speech signals.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
from scipy import signal as sp_signal
import os


class AudioRecorder:
    """
    Record audio signals from microphone.
    """

    def __init__(self, sample_rate=44100, duration=3.0):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Sampling rate in Hz
            duration: Recording duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration

    def record(self, message="Recording..."):
        """
        Record audio from microphone.

        Args:
            message: Message to display during recording

        Returns:
            recording: Recorded audio signal
        """
        print(f"{message} ({self.duration} seconds)")

        # Record audio
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        # Convert to 1D array
        recording = recording.flatten()

        print("Recording complete.")

        return recording

    def save(self, filename, audio):
        """
        Save audio to file.

        Args:
            filename: Output filename
            audio: Audio signal to save
        """
        sf.write(filename, audio, self.sample_rate)
        print(f"Saved: {filename}")


class AudioProcessor:
    """
    Process and prepare audio signals.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize audio processor.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate

    def load_audio(self, filename):
        """
        Load audio from file.

        Args:
            filename: Path to audio file

        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        try:
            audio, sr = sf.read(filename)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio, sr
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None

    def normalize(self, audio):
        """
        Normalize audio signal to [-1, 1] range.

        Args:
            audio: Input audio signal

        Returns:
            normalized_audio: Normalized signal
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized_audio = audio / max_val
        else:
            normalized_audio = audio

        return normalized_audio

    def add_noise(self, audio, snr_db=10):
        """
        Add white Gaussian noise to audio signal.

        Args:
            audio: Clean audio signal
            snr_db: Signal-to-Noise Ratio in dB

        Returns:
            noisy_audio: Audio with added noise
            noise: The noise that was added
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise
        noise = np.sqrt(noise_power) * np.random.randn(len(audio))

        # Add noise to signal
        noisy_audio = audio + noise

        return noisy_audio, noise

    def resample(self, audio, original_sr, target_sr):
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio signal
            original_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            resampled_audio: Resampled signal
        """
        if original_sr == target_sr:
            return audio

        # Calculate resampling ratio
        num_samples = int(len(audio) * target_sr / original_sr)

        # Resample using scipy
        resampled_audio = sp_signal.resample(audio, num_samples)

        return resampled_audio

    def pre_emphasize(self, audio, coeff=0.97):
        """
        Apply pre-emphasis filter to enhance high frequencies.

        Args:
            audio: Input audio signal
            coeff: Pre-emphasis coefficient

        Returns:
            emphasized_audio: Pre-emphasized signal
        """
        emphasized_audio = np.append(audio[0], audio[1:] - coeff * audio[:-1])
        return emphasized_audio

    def compute_fft(self, audio):
        """
        Compute FFT of audio signal.

        Args:
            audio: Input audio signal

        Returns:
            frequencies: Frequency bins
            magnitude: Magnitude spectrum
        """
        # Compute FFT
        fft_result = np.fft.fft(audio)

        # Get magnitude spectrum (single-sided)
        n = len(audio)
        magnitude = np.abs(fft_result[:n // 2])

        # Frequency bins
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)[:n // 2]

        return frequencies, magnitude


def create_dataset_structure(base_dir="data"):
    """
    Create directory structure for storing audio dataset.

    Args:
        base_dir: Base directory for dataset

    Returns:
        paths: Dictionary of directory paths
    """
    paths = {
        'base': base_dir,
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'database': os.path.join(base_dir, 'database'),
        'test': os.path.join(base_dir, 'test'),
        'results': os.path.join(base_dir, 'results')
    }

    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def record_dataset(words=['Hello', 'Estimation', 'Oakland'], num_people=3, sample_rate=44100, duration=3.0):
    """
    Record dataset of words from multiple people.

    Args:
        words: List of words to record
        num_people: Number of different people
        sample_rate: Sampling rate
        duration: Recording duration per word

    Returns:
        recordings: Dictionary of recordings
    """
    recorder = AudioRecorder(sample_rate=sample_rate, duration=duration)
    paths = create_dataset_structure()

    recordings = {}

    for person_id in range(1, num_people + 1):
        print(f"\n--- Person {person_id} ---")
        input(f"Press Enter when Person {person_id} is ready to record...")

        for word in words:
            key = f"{word}_person{person_id}"
            message = f"Say '{word}'"

            # Record
            audio = recorder.record(message)

            # Save
            filename = os.path.join(paths['raw'], f"{key}.wav")
            recorder.save(filename, audio)

            recordings[key] = audio

    return recordings, paths
