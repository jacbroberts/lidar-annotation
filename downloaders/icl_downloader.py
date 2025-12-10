"""
ICL-NUIM Dataset Downloader

ICL-NUIM is a synthetic RGB-D dataset perfect for SLAM benchmarking.
Much smaller than real datasets (~500MB total).

Dataset: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
"""

from pathlib import Path
from typing import Dict, Any
from .base_downloader import BaseDownloader


class ICLNUIMDownloader(BaseDownloader):
    """Handles downloading and organizing ICL-NUIM dataset"""

    # Base URL for ICL-NUIM dataset
    BASE_URL = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC"

    # Available sequences
    SEQUENCES = {
        'living_room_traj0_frei_png': {
            'url': f'{BASE_URL}/living_room_traj0_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Living room trajectory 0',
            'type': 'living_room',
        },
        'living_room_traj1_frei_png': {
            'url': f'{BASE_URL}/living_room_traj1_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Living room trajectory 1',
            'type': 'living_room',
        },
        'living_room_traj2_frei_png': {
            'url': f'{BASE_URL}/living_room_traj2_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Living room trajectory 2',
            'type': 'living_room',
        },
        'living_room_traj3_frei_png': {
            'url': f'{BASE_URL}/living_room_traj3_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Living room trajectory 3',
            'type': 'living_room',
        },
        'office_room_traj0_frei_png': {
            'url': f'{BASE_URL}/office_room_traj0_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Office room trajectory 0',
            'type': 'office_room',
        },
        'office_room_traj1_frei_png': {
            'url': f'{BASE_URL}/office_room_traj1_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Office room trajectory 1',
            'type': 'office_room',
        },
        'office_room_traj2_frei_png': {
            'url': f'{BASE_URL}/office_room_traj2_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Office room trajectory 2',
            'type': 'office_room',
        },
        'office_room_traj3_frei_png': {
            'url': f'{BASE_URL}/office_room_traj3_frei_png.tar.gz',
            'size': '~250MB',
            'description': 'Office room trajectory 3',
            'type': 'office_room',
        },
    }

    def __init__(self, root_dir: str, config: Dict[str, Any], download_options: Dict[str, Any]):
        """
        Initialize ICL-NUIM downloader

        Args:
            root_dir: Root directory to store the dataset
            config: Dataset-specific configuration
            download_options: Download options
        """
        super().__init__(root_dir, config, download_options)

        # Get sequences to download from config
        self.sequences_to_download = config.get('sequences', [
            'living_room_traj0_frei_png',
            'living_room_traj1_frei_png',
        ])

    def download_dataset(self):
        """Download specified sequences of the ICL-NUIM dataset"""
        print(f"\n{'='*60}")
        print("ICL-NUIM Dataset Download")
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

            url = seq_info['url']
            filename = url.split('/')[-1]
            dest_path = self.raw_dir / filename

            # Download
            self.download_file(url, dest_path)

            # Extract if requested
            if self.download_options.get('extract_after_download', True):
                self.extract_archive(dest_path, self.root_dir)

        print("\n" + "="*60)
        print("ICL-NUIM Dataset Download Complete!")
        print("="*60)

        # Print dataset organization
        self.print_dataset_info()

    def print_dataset_info(self):
        """Print information about downloaded sequences"""
        print("\nDataset Organization:")
        print(f"Root: {self.root_dir}")

        # List downloaded sequences
        for seq_dir in sorted(self.root_dir.glob('*_traj*_frei_png')):
            if seq_dir.is_dir():
                print(f"\n  {seq_dir.name}:")

                # Count RGB and depth images
                rgb_files = list(seq_dir.glob('*.png'))
                depth_files = list(seq_dir.glob('*.depth'))

                if rgb_files:
                    print(f"    RGB images: {len(rgb_files)}")

                if depth_files:
                    print(f"    Depth images: {len(depth_files)}")

                # Check for groundtruth
                groundtruth = seq_dir / 'livingRoom.gt.freiburg'
                if not groundtruth.exists():
                    groundtruth = seq_dir / 'traj0.gt.freiburg'
                if not groundtruth.exists():
                    groundtruth = seq_dir / 'traj1.gt.freiburg'

                if groundtruth.exists():
                    print(f"    Groundtruth: available")

    def list_available_sequences(self):
        """List all available sequences"""
        print("\n" + "="*60)
        print("Available ICL-NUIM Sequences")
        print("="*60)

        for key, info in self.SEQUENCES.items():
            print(f"\n{key}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Type: {info['type']}")

        print("\n" + "="*60)


def main():
    """Test the ICL-NUIM downloader"""
    import argparse

    parser = argparse.ArgumentParser(description='Download ICL-NUIM dataset')
    parser.add_argument('--root_dir', type=str, default='./data/icl_nuim',
                        help='Root directory to store dataset')
    parser.add_argument('--list', action='store_true',
                        help='List available sequences')
    parser.add_argument('--sequences', nargs='+',
                        help='Sequences to download')

    args = parser.parse_args()

    # Create config
    config = {
        'sequences': args.sequences or [
            'living_room_traj0_frei_png',
            'living_room_traj1_frei_png',
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

    downloader = ICLNUIMDownloader(args.root_dir, config, download_options)

    if args.list:
        downloader.list_available_sequences()
    else:
        downloader.download_dataset()


if __name__ == '__main__':
    main()
