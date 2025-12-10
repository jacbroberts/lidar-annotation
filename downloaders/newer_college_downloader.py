"""
Newer College Dataset Downloader

Newer College Dataset contains handheld 3D lidar, stereo camera, and IMU data.
Medium size dataset (~5-10GB).

Dataset: https://ori-drs.github.io/newer-college-dataset/
"""

from pathlib import Path
from typing import Dict, Any
from .base_downloader import BaseDownloader


class NewerCollegeDownloader(BaseDownloader):
    """Handles downloading and organizing Newer College dataset"""

    # Base URL for Newer College dataset
    BASE_URL = "https://ori-drs.github.io/newer-college-dataset"
    DOWNLOAD_PAGE = "https://ori-drs.github.io/newer-college-dataset/download/"
    CONTACT_EMAIL = "newercollegedataset@robots.ox.ac.uk"

    # Available sequences (Note: These require manual download from the website)
    SEQUENCES = {
        '01_short_experiment': {
            'description': 'Short handheld experiment',
            'size': '~2GB',
            'duration': '~5 minutes',
            'sensors': ['Ouster OS-1 LiDAR', 'Intel RealSense D435i', 'IMU'],
        },
        '02_long_experiment': {
            'description': 'Long handheld experiment',
            'size': '~8GB',
            'duration': '~20 minutes',
            'sensors': ['Ouster OS-1 LiDAR', 'Intel RealSense D435i', 'IMU'],
        },
        'cloister': {
            'description': 'Cloister sequence',
            'size': '~3GB',
            'duration': '~10 minutes',
            'sensors': ['Ouster OS-1 LiDAR', 'Intel RealSense D435i', 'IMU'],
        },
        'math_easy': {
            'description': 'Math institute - easy trajectory',
            'size': '~2GB',
            'duration': '~7 minutes',
            'sensors': ['Ouster OS-1 LiDAR', 'Intel RealSense D435i', 'IMU'],
        },
        'math_hard': {
            'description': 'Math institute - hard trajectory',
            'size': '~3GB',
            'duration': '~9 minutes',
            'sensors': ['Ouster OS-1 LiDAR', 'Intel RealSense D435i', 'IMU'],
        },
    }

    def __init__(self, root_dir: str, config: Dict[str, Any], download_options: Dict[str, Any]):
        """
        Initialize Newer College downloader

        Args:
            root_dir: Root directory to store the dataset
            config: Dataset-specific configuration
            download_options: Download options
        """
        super().__init__(root_dir, config, download_options)

        # Get sequences to download from config
        self.sequences_to_download = config.get('sequences', [
            '01_short_experiment',
        ])

    def download_dataset(self):
        """Download specified sequences of the Newer College dataset"""
        print(f"\n{'='*70}")
        print("Newer College Dataset Download Information")
        print(f"{'='*70}")
        print("\nIMPORTANT: The Newer College Dataset requires manual download.")
        print("The dataset is provided in rosbag format and must be obtained from")
        print("the official website.\n")
        print(f"Download Page: {self.DOWNLOAD_PAGE}")
        print(f"Contact: {self.CONTACT_EMAIL}\n")
        print("="*70)
        print("\nInstructions:")
        print("1. Visit the download page above")
        print("2. Fill out the data access form")
        print("3. Download the requested sequences")
        print(f"4. Extract the downloaded files to: {self.root_dir}")
        print("5. Ensure the directory structure matches the expected format\n")

        print("="*70)
        print("Sequences requested:")
        print("="*70)

        for seq_key in self.sequences_to_download:
            if seq_key not in self.SEQUENCES:
                print(f"Warning: Unknown sequence '{seq_key}', skipping...")
                continue

            seq_info = self.SEQUENCES[seq_key]
            print(f"\n{seq_key}:")
            print(f"  Description: {seq_info['description']}")
            print(f"  Size: {seq_info['size']}")
            print(f"  Duration: {seq_info['duration']}")
            print(f"  Sensors: {', '.join(seq_info['sensors'])}")

        print("\n" + "="*70)
        print("Expected Directory Structure:")
        print("="*70)
        print(f"{self.root_dir}/")
        print("  +-- 01_short_experiment/")
        print("  |   +-- ouster_scan/")
        print("  |   +-- stereo/")
        print("  |   +-- imu/")
        print("  |   +-- ground_truth/")
        print("  +-- 02_long_experiment/")
        print("      +-- ...")
        print("\n" + "="*70)

        # Check if any data already exists
        existing_sequences = []
        for seq_dir in sorted(self.root_dir.glob('*experiment*')):
            if seq_dir.is_dir():
                existing_sequences.append(seq_dir.name)

        if existing_sequences:
            print("\nFound existing sequences:")
            for seq in existing_sequences:
                print(f"  ✓ {seq}")
            print("\nDataset organization:")
            self.print_dataset_info()
        else:
            print("\nNo sequences found yet. Please download from the website above.")
            print("="*70)

    def print_dataset_info(self):
        """Print information about downloaded sequences"""
        print("\nDataset Organization:")
        print(f"Root: {self.root_dir}")

        # List downloaded sequences
        for seq_dir in sorted(self.root_dir.glob('0*_*_experiment')):
            if seq_dir.is_dir():
                print(f"\n  {seq_dir.name}:")

                # Count lidar scans
                lidar_dir = seq_dir / 'ouster_scan'
                if lidar_dir.exists():
                    n_scans = len(list(lidar_dir.glob('*.bin')))
                    print(f"    Lidar scans: {n_scans}")

                # Count stereo images
                left_dir = seq_dir / 'stereo' / 'left'
                right_dir = seq_dir / 'stereo' / 'right'

                if left_dir and left_dir.exists():
                    n_left = len(list(left_dir.glob('*.png')))
                    print(f"    Left camera images: {n_left}")

                if right_dir and right_dir.exists():
                    n_right = len(list(right_dir.glob('*.png')))
                    print(f"    Right camera images: {n_right}")

                # Check for IMU data
                imu_file = seq_dir / 'imu' / 'data.csv'
                if imu_file.exists():
                    print(f"    IMU data: available")

                # Check for groundtruth
                gt_file = seq_dir / 'ground_truth' / 'registered_poses.csv'
                if gt_file.exists():
                    print(f"    Groundtruth: available")

    def list_available_sequences(self):
        """List all available sequences"""
        print("\n" + "="*60)
        print("Available Newer College Sequences")
        print("="*60)

        for key, info in self.SEQUENCES.items():
            print(f"\n{key}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Duration: {info['duration']}")

        print("\n" + "="*60)


def main():
    """Test the Newer College downloader"""
    import argparse

    parser = argparse.ArgumentParser(description='Download Newer College dataset')
    parser.add_argument('--root_dir', type=str, default='./data/newer_college',
                        help='Root directory to store dataset')
    parser.add_argument('--list', action='store_true',
                        help='List available sequences')
    parser.add_argument('--sequences', nargs='+',
                        help='Sequences to download')

    args = parser.parse_args()

    # Create config
    config = {
        'sequences': args.sequences or ['01_short_experiment']
    }

    download_options = {
        'chunk_size': 1024*1024,
        'max_retries': 5,
        'retry_delay': 5,
        'timeout': {'connect': 10, 'read': 300},
        'resume_downloads': True,
        'extract_after_download': True,
    }

    downloader = NewerCollegeDownloader(args.root_dir, config, download_options)

    if args.list:
        downloader.list_available_sequences()
    else:
        downloader.download_dataset()


if __name__ == '__main__':
    main()
