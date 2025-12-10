"""
TUM RGB-D Dataset Downloader

TUM RGB-D Dataset is a collection of RGB-D sequences for visual odometry
and SLAM evaluation. Much smaller than KITTI (~1-2GB per sequence).

Dataset: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""

from pathlib import Path
from typing import Dict, Any
from .base_downloader import BaseDownloader


class TUMDownloader(BaseDownloader):
    """Handles downloading and organizing TUM RGB-D dataset"""

    # TUM RGB-D dataset URLs base
    BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg{}/rgbd_dataset_freiburg{}_{}."

    # Available sequences with their details
    SEQUENCES = {
        'freiburg1/rgbd_dataset_freiburg1_xyz': {
            'freiburg': '1',
            'name': 'xyz',
            'size': '~700MB',
            'description': 'Simple XYZ motion',
        },
        'freiburg1/rgbd_dataset_freiburg1_desk': {
            'freiburg': '1',
            'name': 'desk',
            'size': '~1.2GB',
            'description': 'Desk scene with loop closure',
        },
        'freiburg1/rgbd_dataset_freiburg1_desk2': {
            'freiburg': '1',
            'name': 'desk2',
            'size': '~1.0GB',
            'description': 'Desk scene validation',
        },
        'freiburg1/rgbd_dataset_freiburg1_room': {
            'freiburg': '1',
            'name': 'room',
            'size': '~1.4GB',
            'description': 'Large room loop',
        },
        'freiburg2/rgbd_dataset_freiburg2_xyz': {
            'freiburg': '2',
            'name': 'xyz',
            'size': '~1.0GB',
            'description': 'XYZ motion with faster camera',
        },
        'freiburg2/rgbd_dataset_freiburg2_desk': {
            'freiburg': '2',
            'name': 'desk',
            'size': '~1.5GB',
            'description': 'Desk scene with faster motion',
        },
        'freiburg3/rgbd_dataset_freiburg3_long_office_household': {
            'freiburg': '3',
            'name': 'long_office_household',
            'size': '~2.6GB',
            'description': 'Long office and household sequence',
        },
    }

    def __init__(self, root_dir: str, config: Dict[str, Any], download_options: Dict[str, Any]):
        """
        Initialize TUM RGB-D downloader

        Args:
            root_dir: Root directory to store the dataset
            config: Dataset-specific configuration
            download_options: Download options
        """
        super().__init__(root_dir, config, download_options)

        # Get sequences to download from config
        self.sequences_to_download = config.get('sequences', [
            'freiburg1/rgbd_dataset_freiburg1_xyz',
            'freiburg1/rgbd_dataset_freiburg1_desk',
        ])

    def _get_download_url(self, sequence_key: str) -> str:
        """
        Get download URL for a sequence

        Args:
            sequence_key: Sequence identifier

        Returns:
            Download URL for the sequence
        """
        if sequence_key not in self.SEQUENCES:
            raise ValueError(f"Unknown sequence: {sequence_key}")

        seq_info = self.SEQUENCES[sequence_key]
        freiburg_num = seq_info['freiburg']
        seq_name = seq_info['name']

        # TUM provides .tgz archives
        url = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg{freiburg_num}/rgbd_dataset_freiburg{freiburg_num}_{seq_name}.tgz"
        return url

    def download_dataset(self):
        """Download specified sequences of the TUM RGB-D dataset"""
        print(f"\n{'='*60}")
        print("TUM RGB-D Dataset Download")
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

            url = self._get_download_url(seq_key)
            filename = url.split('/')[-1]
            dest_path = self.raw_dir / filename

            # Download
            self.download_file(url, dest_path)

            # Extract if requested
            if self.download_options.get('extract_after_download', True):
                self.extract_archive(dest_path, self.root_dir)

        print("\n" + "="*60)
        print("TUM RGB-D Dataset Download Complete!")
        print("="*60)

        # Print dataset organization
        self.print_dataset_info()

    def print_dataset_info(self):
        """Print information about downloaded sequences"""
        print("\nDataset Organization:")
        print(f"Root: {self.root_dir}")

        # List downloaded sequences
        for seq_dir in sorted(self.root_dir.glob('rgbd_dataset_*')):
            if seq_dir.is_dir():
                print(f"\n  {seq_dir.name}:")

                # Count RGB and depth images
                rgb_dir = seq_dir / 'rgb'
                depth_dir = seq_dir / 'depth'

                if rgb_dir.exists():
                    n_rgb = len(list(rgb_dir.glob('*.png')))
                    print(f"    RGB images: {n_rgb}")

                if depth_dir.exists():
                    n_depth = len(list(depth_dir.glob('*.png')))
                    print(f"    Depth images: {n_depth}")

                # Check for groundtruth
                groundtruth = seq_dir / 'groundtruth.txt'
                if groundtruth.exists():
                    print(f"    Groundtruth: available")

    def get_sequence_info(self, sequence_key: str) -> Dict[str, Any]:
        """
        Get information about a specific sequence

        Args:
            sequence_key: Sequence identifier

        Returns:
            Dictionary with sequence information
        """
        if sequence_key not in self.SEQUENCES:
            raise ValueError(f"Unknown sequence: {sequence_key}")

        return self.SEQUENCES[sequence_key]

    def list_available_sequences(self):
        """List all available sequences"""
        print("\n" + "="*60)
        print("Available TUM RGB-D Sequences")
        print("="*60)

        for key, info in self.SEQUENCES.items():
            print(f"\n{key}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")

        print("\n" + "="*60)


def main():
    """Test the TUM downloader"""
    import argparse

    parser = argparse.ArgumentParser(description='Download TUM RGB-D dataset')
    parser.add_argument('--root_dir', type=str, default='./data/tum_rgbd',
                        help='Root directory to store dataset')
    parser.add_argument('--list', action='store_true',
                        help='List available sequences')
    parser.add_argument('--sequences', nargs='+',
                        help='Sequences to download')

    args = parser.parse_args()

    # Create config
    config = {
        'sequences': args.sequences or [
            'freiburg1/rgbd_dataset_freiburg1_xyz',
            'freiburg1/rgbd_dataset_freiburg1_desk',
        ]
    }

    download_options = {
        'chunk_size': 1024*1024,
        'max_retries': 5,
        'retry_delay': 5,
        'timeout': {'connect': 10, 'read': 300},
        'resume_downloads': True,
        'extract_after_download': True,
    }

    downloader = TUMDownloader(args.root_dir, config, download_options)

    if args.list:
        downloader.list_available_sequences()
    else:
        downloader.download_dataset()


if __name__ == '__main__':
    main()
