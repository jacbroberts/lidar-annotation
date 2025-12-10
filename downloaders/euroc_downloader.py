"""
EuRoC MAV Dataset Downloader

EuRoC MAV Dataset contains stereo images, IMU data from a drone/MAV.
Medium size dataset (~5-10GB per sequence).

Dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
"""

from pathlib import Path
from typing import Dict, Any
from .base_downloader import BaseDownloader


class EuRoCDownloader(BaseDownloader):
    """Handles downloading and organizing EuRoC MAV dataset"""

    # Base URL for EuRoC dataset
    BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

    # Available sequences
    SEQUENCES = {
        'MH_01_easy': {
            'url': f'{BASE_URL}/machine_hall/MH_01_easy/MH_01_easy.zip',
            'size': '~3.5GB',
            'description': 'Machine Hall - Easy',
            'location': 'machine_hall',
            'difficulty': 'easy',
        },
        'MH_02_easy': {
            'url': f'{BASE_URL}/machine_hall/MH_02_easy/MH_02_easy.zip',
            'size': '~4.5GB',
            'description': 'Machine Hall - Easy 2',
            'location': 'machine_hall',
            'difficulty': 'easy',
        },
        'MH_03_medium': {
            'url': f'{BASE_URL}/machine_hall/MH_03_medium/MH_03_medium.zip',
            'size': '~4.0GB',
            'description': 'Machine Hall - Medium',
            'location': 'machine_hall',
            'difficulty': 'medium',
        },
        'MH_04_difficult': {
            'url': f'{BASE_URL}/machine_hall/MH_04_difficult/MH_04_difficult.zip',
            'size': '~3.5GB',
            'description': 'Machine Hall - Difficult',
            'location': 'machine_hall',
            'difficulty': 'difficult',
        },
        'MH_05_difficult': {
            'url': f'{BASE_URL}/machine_hall/MH_05_difficult/MH_05_difficult.zip',
            'size': '~3.8GB',
            'description': 'Machine Hall - Difficult 2',
            'location': 'machine_hall',
            'difficulty': 'difficult',
        },
        'V1_01_easy': {
            'url': f'{BASE_URL}/vicon_room1/V1_01_easy/V1_01_easy.zip',
            'size': '~1.8GB',
            'description': 'Vicon Room 1 - Easy',
            'location': 'vicon_room1',
            'difficulty': 'easy',
        },
        'V1_02_medium': {
            'url': f'{BASE_URL}/vicon_room1/V1_02_medium/V1_02_medium.zip',
            'size': '~3.5GB',
            'description': 'Vicon Room 1 - Medium',
            'location': 'vicon_room1',
            'difficulty': 'medium',
        },
        'V1_03_difficult': {
            'url': f'{BASE_URL}/vicon_room1/V1_03_difficult/V1_03_difficult.zip',
            'size': '~3.7GB',
            'description': 'Vicon Room 1 - Difficult',
            'location': 'vicon_room1',
            'difficulty': 'difficult',
        },
        'V2_01_easy': {
            'url': f'{BASE_URL}/vicon_room2/V2_01_easy/V2_01_easy.zip',
            'size': '~2.3GB',
            'description': 'Vicon Room 2 - Easy',
            'location': 'vicon_room2',
            'difficulty': 'easy',
        },
        'V2_02_medium': {
            'url': f'{BASE_URL}/vicon_room2/V2_02_medium/V2_02_medium.zip',
            'size': '~4.0GB',
            'description': 'Vicon Room 2 - Medium',
            'location': 'vicon_room2',
            'difficulty': 'medium',
        },
        'V2_03_difficult': {
            'url': f'{BASE_URL}/vicon_room2/V2_03_difficult/V2_03_difficult.zip',
            'size': '~3.2GB',
            'description': 'Vicon Room 2 - Difficult',
            'location': 'vicon_room2',
            'difficulty': 'difficult',
        },
    }

    def __init__(self, root_dir: str, config: Dict[str, Any], download_options: Dict[str, Any]):
        """
        Initialize EuRoC downloader

        Args:
            root_dir: Root directory to store the dataset
            config: Dataset-specific configuration
            download_options: Download options
        """
        super().__init__(root_dir, config, download_options)

        # Get sequences to download from config
        self.sequences_to_download = config.get('sequences', [
            'MH_01_easy',
        ])

    def download_dataset(self):
        """Download specified sequences of the EuRoC dataset"""
        print(f"\n{'='*60}")
        print("EuRoC MAV Dataset Download")
        print(f"{'='*60}")
        print(f"Sequences to download: {len(self.sequences_to_download)}")

        for seq_key in self.sequences_to_download:
            if seq_key not in self.SEQUENCES:
                print(f"Warning: Unknown sequence '{seq_key}', skipping...")
                continue

            seq_info = self.SEQUENCES[seq_key]
            print(f"\n--- Sequence: {seq_key} ---")
            print(f"Description: {seq_info['description']}")
            print(f"Size: {seq_info['size']}")
            print(f"Difficulty: {seq_info['difficulty']}")

            url = seq_info['url']
            filename = url.split('/')[-1]
            dest_path = self.raw_dir / filename

            # Download
            self.download_file(url, dest_path)

            # Extract if requested
            if self.download_options.get('extract_after_download', True):
                self.extract_archive(dest_path, self.root_dir)

        print("\n" + "="*60)
        print("EuRoC MAV Dataset Download Complete!")
        print("="*60)

        # Print dataset organization
        self.print_dataset_info()

    def print_dataset_info(self):
        """Print information about downloaded sequences"""
        print("\nDataset Organization:")
        print(f"Root: {self.root_dir}")

        # List downloaded sequences
        for seq_dir in sorted(self.root_dir.glob('*_0*')):
            if seq_dir.is_dir():
                print(f"\n  {seq_dir.name}:")

                # Count stereo images
                cam0_dir = seq_dir / 'mav0' / 'cam0' / 'data'
                cam1_dir = seq_dir / 'mav0' / 'cam1' / 'data'

                if cam0_dir.exists():
                    n_cam0 = len(list(cam0_dir.glob('*.png')))
                    print(f"    Camera 0 images: {n_cam0}")

                if cam1_dir.exists():
                    n_cam1 = len(list(cam1_dir.glob('*.png')))
                    print(f"    Camera 1 images: {n_cam1}")

                # Check for IMU data
                imu_file = seq_dir / 'mav0' / 'imu0' / 'data.csv'
                if imu_file.exists():
                    print(f"    IMU data: available")

                # Check for groundtruth
                gt_file = seq_dir / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
                if gt_file.exists():
                    print(f"    Groundtruth: available")

    def list_available_sequences(self):
        """List all available sequences"""
        print("\n" + "="*60)
        print("Available EuRoC MAV Sequences")
        print("="*60)

        for key, info in self.SEQUENCES.items():
            print(f"\n{key}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Location: {info['location']}")
            print(f"  Difficulty: {info['difficulty']}")

        print("\n" + "="*60)


def main():
    """Test the EuRoC downloader"""
    import argparse

    parser = argparse.ArgumentParser(description='Download EuRoC MAV dataset')
    parser.add_argument('--root_dir', type=str, default='./data/euroc',
                        help='Root directory to store dataset')
    parser.add_argument('--list', action='store_true',
                        help='List available sequences')
    parser.add_argument('--sequences', nargs='+',
                        help='Sequences to download')

    args = parser.parse_args()

    # Create config
    config = {
        'sequences': args.sequences or ['MH_01_easy']
    }

    download_options = {
        'chunk_size': 1024*1024,
        'max_retries': 5,
        'retry_delay': 5,
        'timeout': {'connect': 10, 'read': 300},
        'resume_downloads': True,
        'extract_after_download': True,
    }

    downloader = EuRoCDownloader(args.root_dir, config, download_options)

    if args.list:
        downloader.list_available_sequences()
    else:
        downloader.download_dataset()


if __name__ == '__main__':
    main()
