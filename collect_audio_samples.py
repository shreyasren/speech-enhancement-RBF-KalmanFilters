"""
Audio Sample Collection Utility

This script helps organize and prepare 40 real speech audio samples for experiments.
Supports:
- Manual file organization (copy your own WAV files)
- Automatic download from internet sources (optional)
- Format validation and conversion

Usage:
    1. Manual: Place your WAV files in data/raw/ with naming convention:
       - internet_sample_01.wav, internet_sample_02.wav, ... (20 files)
       - entertainment_sample_01.wav, entertainment_sample_02.wav, ... (20 files)

    2. Automatic: Run this script to download from public datasets (requires internet)

    3. Validation: Run with --validate to check all files are properly formatted
"""

import os
import sys
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class AudioSampleCollector:
    """
    Utility for collecting and organizing audio samples.
    """

    def __init__(self, data_dir="data"):
        """
        Initialize collector.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.target_sr = 16000  # Standard speech processing sample rate

        # Expected file naming
        self.internet_samples = [f"internet_sample_{i:02d}.wav" for i in range(1, 21)]
        self.entertainment_samples = [f"entertainment_sample_{i:02d}.wav" for i in range(1, 21)]

    def validate_samples(self):
        """
        Validate that all required samples exist and are properly formatted.

        Returns:
            validation_report: Dictionary with validation results
        """
        print("\n" + "=" * 70)
        print("VALIDATING AUDIO SAMPLES")
        print("=" * 70)

        report = {
            'internet': {'found': [], 'missing': [], 'invalid': []},
            'entertainment': {'found': [], 'missing': [], 'invalid': []},
            'total_found': 0,
            'total_missing': 0,
            'total_invalid': 0
        }

        # Check internet samples
        print("\nChecking Internet Samples (20 expected):")
        for filename in self.internet_samples:
            filepath = self.raw_dir / filename

            if filepath.exists():
                try:
                    audio, sr = sf.read(str(filepath))
                    duration = len(audio) / sr

                    print(f"  ✓ {filename}: {sr} Hz, {duration:.2f}s")
                    report['internet']['found'].append(filename)
                    report['total_found'] += 1

                except Exception as e:
                    print(f"  ✗ {filename}: INVALID - {e}")
                    report['internet']['invalid'].append(filename)
                    report['total_invalid'] += 1
            else:
                print(f"  ✗ {filename}: MISSING")
                report['internet']['missing'].append(filename)
                report['total_missing'] += 1

        # Check entertainment samples
        print("\nChecking Entertainment Samples (20 expected):")
        for filename in self.entertainment_samples:
            filepath = self.raw_dir / filename

            if filepath.exists():
                try:
                    audio, sr = sf.read(str(filepath))
                    duration = len(audio) / sr

                    print(f"  ✓ {filename}: {sr} Hz, {duration:.2f}s")
                    report['entertainment']['found'].append(filename)
                    report['total_found'] += 1

                except Exception as e:
                    print(f"  ✗ {filename}: INVALID - {e}")
                    report['entertainment']['invalid'].append(filename)
                    report['total_invalid'] += 1
            else:
                print(f"  ✗ {filename}: MISSING")
                report['entertainment']['missing'].append(filename)
                report['total_missing'] += 1

        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total samples found: {report['total_found']}/40")
        print(f"Total samples missing: {report['total_missing']}/40")
        print(f"Total invalid files: {report['total_invalid']}")

        if report['total_found'] == 40:
            print("\n✓ All 40 samples are present and valid!")
        else:
            print(f"\n✗ Missing {report['total_missing']} samples")
            print("\nTo complete the dataset:")
            print("1. Add WAV files to: data/raw/")
            print("2. Use naming convention: internet_sample_XX.wav or entertainment_sample_XX.wav")
            print("3. Run this script again with --validate")

        return report

    def convert_to_standard_format(self, input_file, output_file, target_sr=16000):
        """
        Convert audio file to standard format (16kHz mono WAV).

        Args:
            input_file: Input audio file path
            output_file: Output WAV file path
            target_sr: Target sample rate

        Returns:
            success: True if successful
        """
        try:
            # Load audio
            audio, sr = sf.read(str(input_file))

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != target_sr:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
                sr = target_sr

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Leave headroom

            # Save
            sf.write(str(output_file), audio, sr)

            print(f"  ✓ Converted: {input_file.name} -> {output_file.name}")
            return True

        except Exception as e:
            print(f"  ✗ Error converting {input_file.name}: {e}")
            return False

    def batch_convert_directory(self, input_dir, category='internet'):
        """
        Batch convert all audio files in a directory to standard format.

        Args:
            input_dir: Directory containing audio files to convert
            category: 'internet' or 'entertainment'

        Returns:
            num_converted: Number of files successfully converted
        """
        print(f"\nBatch converting {category} samples from: {input_dir}")

        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  ✗ Directory not found: {input_dir}")
            return 0

        # Supported formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        # Find audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))

        audio_files = sorted(audio_files)[:20]  # Take first 20

        if not audio_files:
            print(f"  ✗ No audio files found in {input_dir}")
            return 0

        print(f"  Found {len(audio_files)} audio files")

        # Convert each file
        num_converted = 0
        for i, audio_file in enumerate(audio_files, 1):
            output_filename = f"{category}_sample_{i:02d}.wav"
            output_path = self.raw_dir / output_filename

            if self.convert_to_standard_format(audio_file, output_path, self.target_sr):
                num_converted += 1

        print(f"\n  ✓ Successfully converted {num_converted}/{len(audio_files)} files")
        return num_converted

    def download_sample_dataset(self):
        """
        Download sample speech dataset from public sources.

        Note: This is a template. Actual implementation would require
        specific dataset URLs and download permissions.
        """
        print("\n" + "=" * 70)
        print("SAMPLE DATASET DOWNLOAD")
        print("=" * 70)
        print("\nThis feature requires manual implementation based on your data sources.")
        print("\nSuggested public speech datasets:")
        print("  1. LibriSpeech: https://www.openslr.org/12/")
        print("  2. Common Voice: https://commonvoice.mozilla.org/")
        print("  3. VCTK Corpus: https://datashare.ed.ac.uk/handle/10283/3443")
        print("  4. TIMIT: https://catalog.ldc.upenn.edu/LDC93S1")
        print("\nFor entertainment samples:")
        print("  - Movie clips (ensure proper licensing)")
        print("  - YouTube videos (with proper attribution)")
        print("  - Podcast excerpts")
        print("\nPlease download manually and use --convert option.")

    def create_manifest(self):
        """
        Create a manifest file listing all available samples.

        Returns:
            manifest_path: Path to created manifest file
        """
        manifest_path = self.data_dir / 'sample_manifest.txt'

        with open(manifest_path, 'w') as f:
            f.write("Audio Sample Manifest\n")
            f.write("=" * 70 + "\n\n")

            # Internet samples
            f.write("Internet Samples:\n")
            for filename in self.internet_samples:
                filepath = self.raw_dir / filename
                if filepath.exists():
                    audio, sr = sf.read(str(filepath))
                    duration = len(audio) / sr
                    f.write(f"  ✓ {filename}: {sr} Hz, {duration:.2f}s\n")
                else:
                    f.write(f"  ✗ {filename}: MISSING\n")

            f.write("\n")

            # Entertainment samples
            f.write("Entertainment Samples:\n")
            for filename in self.entertainment_samples:
                filepath = self.raw_dir / filename
                if filepath.exists():
                    audio, sr = sf.read(str(filepath))
                    duration = len(audio) / sr
                    f.write(f"  ✓ {filename}: {sr} Hz, {duration:.2f}s\n")
                else:
                    f.write(f"  ✗ {filename}: MISSING\n")

        print(f"\n✓ Manifest created: {manifest_path}")
        return manifest_path

    def generate_sample_filenames(self):
        """
        Print sample filenames for manual organization.
        """
        print("\n" + "=" * 70)
        print("EXPECTED FILENAME STRUCTURE")
        print("=" * 70)

        print("\nInternet Samples (20 files):")
        for filename in self.internet_samples:
            print(f"  {filename}")

        print("\nEntertainment Samples (20 files):")
        for filename in self.entertainment_samples:
            print(f"  {filename}")

        print("\nAll files should be placed in: data/raw/")
        print("Format: 16kHz mono WAV (will be auto-converted if different)")


def main():
    """
    Main execution.
    """
    parser = argparse.ArgumentParser(
        description='Audio Sample Collection Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate existing samples
  python3 collect_audio_samples.py --validate

  # Convert files from a directory
  python3 collect_audio_samples.py --convert /path/to/internet/samples --category internet
  python3 collect_audio_samples.py --convert /path/to/entertainment/samples --category entertainment

  # Show expected filename structure
  python3 collect_audio_samples.py --show-filenames

  # Create manifest
  python3 collect_audio_samples.py --manifest
        """
    )

    parser.add_argument('--validate', action='store_true',
                       help='Validate existing audio samples')
    parser.add_argument('--convert', type=str,
                       help='Convert audio files from directory to standard format')
    parser.add_argument('--category', type=str, choices=['internet', 'entertainment'],
                       help='Category for converted files (required with --convert)')
    parser.add_argument('--download', action='store_true',
                       help='Show instructions for downloading sample datasets')
    parser.add_argument('--manifest', action='store_true',
                       help='Create a manifest file of all samples')
    parser.add_argument('--show-filenames', action='store_true',
                       help='Show expected filename structure')

    args = parser.parse_args()

    collector = AudioSampleCollector()

    if args.validate:
        collector.validate_samples()

    elif args.convert:
        if not args.category:
            print("Error: --category required with --convert")
            print("Use: --category internet  or  --category entertainment")
            sys.exit(1)

        collector.batch_convert_directory(args.convert, args.category)

        # Validate after conversion
        print("\nRunning validation...")
        collector.validate_samples()

    elif args.download:
        collector.download_sample_dataset()

    elif args.manifest:
        collector.create_manifest()

    elif args.show_filenames:
        collector.generate_sample_filenames()

    else:
        # Default: show help and validate
        parser.print_help()
        print("\n")
        collector.validate_samples()


if __name__ == "__main__":
    main()
