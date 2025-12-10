"""
KITTI Odometry Dataset Download and Organization Script

This script automates the download and extraction of the KITTI Odometry dataset.
Dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse
import time


class KITTIDownloader:
    """Handles downloading and organizing KITTI Odometry dataset"""

    # KITTI Odometry dataset URLs
    URLS = {
        'odometry_color': [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip',
        ],
        'odometry_velodyne': [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip',
        ],
        'odometry_calib': [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip',
        ],
        'odometry_poses': [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip',
        ],
    }

    def __init__(self, root_dir='./data/kitti'):
        """
        Initialize the downloader

        Args:
            root_dir: Root directory to store the dataset
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root_dir / 'raw'
        self.raw_dir.mkdir(exist_ok=True)

        # Create optimized session with retry logic and connection pooling
        self.session = self._create_session()

    def _create_session(self):
        """
        Create a requests session with retry logic and connection pooling
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=10,  # Maximum number of retries
            backoff_factor=1,  # Wait 1, 2, 4, 8, 16... seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Methods to retry
        )

        # Mount adapter with retry strategy for both http and https
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool size
            pool_maxsize=20  # Max pool size
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set timeout defaults
        session.timeout = (10, 300)  # (connect timeout, read timeout) in seconds

        return session

    def download_file(self, url, dest_path, chunk_size=1024*1024):
        """
        Download a file with progress bar, resume capability, and retry logic

        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Size of chunks to download (default: 1MB for better performance)
        """
        # Check if file exists and get its size for resume
        resume_byte_pos = 0
        temp_path = Path(str(dest_path) + '.partial')

        if dest_path.exists():
            print(f"File already exists: {dest_path}")
            return

        if temp_path.exists():
            resume_byte_pos = temp_path.stat().st_size
            print(f"Resuming download from byte {resume_byte_pos}")

        # Prepare headers for resume
        headers = {}
        if resume_byte_pos > 0:
            headers['Range'] = f'bytes={resume_byte_pos}-'

        print(f"Downloading: {url}")

        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=(10, 300)  # (connect, read) timeout
                )
                response.raise_for_status()

                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                if resume_byte_pos > 0:
                    total_size += resume_byte_pos

                # Open file in append mode if resuming, write mode otherwise
                file_mode = 'ab' if resume_byte_pos > 0 else 'wb'

                with open(temp_path, file_mode) as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    initial=resume_byte_pos,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter out keep-alive chunks
                            size = f.write(data)
                            pbar.update(size)

                # Download successful, rename temp file to final name
                temp_path.rename(dest_path)
                print(f"Download complete: {dest_path}")
                return

            except (requests.exceptions.RequestException, IOError) as e:
                if attempt < max_retries - 1:
                    print(f"\nDownload error: {e}")
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(retry_delay)
                    # Update resume position if temp file exists
                    if temp_path.exists():
                        resume_byte_pos = temp_path.stat().st_size
                        headers['Range'] = f'bytes={resume_byte_pos}-'
                else:
                    print(f"\nFailed to download after {max_retries} attempts")
                    raise

    def extract_archive(self, archive_path, extract_to):
        """
        Extract zip or tar archive

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to
        """
        print(f"Extracting: {archive_path}")

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path}")

    def download_dataset(self, components=None):
        """
        Download specified components of the dataset

        Args:
            components: List of components to download. If None, downloads all.
                       Options: 'odometry_color', 'odometry_velodyne',
                               'odometry_calib', 'odometry_poses'
        """
        if components is None:
            components = list(self.URLS.keys())

        for component in components:
            if component not in self.URLS:
                print(f"Unknown component: {component}")
                continue

            for url in self.URLS[component]:
                filename = url.split('/')[-1]
                dest_path = self.raw_dir / filename

                # Download
                self.download_file(url, dest_path)

                # Extract
                self.extract_archive(dest_path, self.root_dir)

        print("\nDownload and extraction complete!")

    def organize_dataset(self):
        """
        Organize the extracted dataset into a clean structure
        """
        print("\nOrganizing dataset...")

        # Create organized structure
        sequences_dir = self.root_dir / 'sequences'
        dataset_dir = self.root_dir / 'dataset'

        if sequences_dir.exists():
            print(f"Dataset already organized at: {sequences_dir}")
            # Print dataset statistics
            self.print_dataset_stats()
        elif dataset_dir.exists():
            # Move sequences and poses from dataset/ to root
            print(f"Found dataset directory, moving contents to root...")

            import shutil

            # Move sequences directory
            dataset_sequences = dataset_dir / 'sequences'
            if dataset_sequences.exists():
                if sequences_dir.exists():
                    shutil.rmtree(sequences_dir)
                shutil.move(str(dataset_sequences), str(sequences_dir))
                print(f"  Moved sequences to: {sequences_dir}")

            # Move poses directory
            dataset_poses = dataset_dir / 'poses'
            poses_dir = self.root_dir / 'poses'
            if dataset_poses.exists():
                if poses_dir.exists():
                    shutil.rmtree(poses_dir)
                shutil.move(str(dataset_poses), str(poses_dir))
                print(f"  Moved poses to: {poses_dir}")

            # Remove empty dataset directory
            if dataset_dir.exists():
                try:
                    dataset_dir.rmdir()
                    print(f"  Removed empty dataset directory")
                except:
                    print(f"  Note: dataset directory not empty, keeping it")

            print("Dataset organization complete!")
            self.print_dataset_stats()
        else:
            print("Warning: sequences directory not found after extraction.")
            print("The dataset might not have been extracted properly.")

    def print_dataset_stats(self):
        """Print statistics about the downloaded dataset"""
        sequences_dir = self.root_dir / 'sequences'

        if not sequences_dir.exists():
            print("No sequences directory found.")
            return

        print("\n" + "="*50)
        print("KITTI Odometry Dataset Statistics")
        print("="*50)

        sequences = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])

        for seq_dir in sequences:
            velodyne_dir = seq_dir / 'velodyne'
            image_dir = seq_dir / 'image_2'

            if velodyne_dir.exists():
                n_scans = len(list(velodyne_dir.glob('*.bin')))
                print(f"Sequence {seq_dir.name}: {n_scans} point cloud scans")

            if image_dir.exists():
                n_images = len(list(image_dir.glob('*.png')))
                print(f"  - {n_images} images")

        print("="*50)

    def cleanup_temp_files(self):
        """Clean up any partial download files"""
        print("\nCleaning up temporary files...")
        partial_files = list(self.raw_dir.glob('*.partial'))
        for pf in partial_files:
            print(f"  Removing: {pf}")
            pf.unlink()

    def __del__(self):
        """Cleanup on object destruction"""
        if hasattr(self, 'session'):
            self.session.close()


def main():
    parser = argparse.ArgumentParser(
        description='Download and organize KITTI Odometry dataset'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='./data/kitti',
        help='Root directory to store dataset'
    )
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['odometry_color', 'odometry_velodyne', 'odometry_calib', 'odometry_poses', 'all'],
        default=['all'],
        help='Components to download'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download and only organize existing files'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up partial download files'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = KITTIDownloader(root_dir=args.root_dir)

    # Cleanup if requested
    if args.cleanup:
        downloader.cleanup_temp_files()
        return

    # Download dataset
    if not args.skip_download:
        components = None if 'all' in args.components else args.components
        downloader.download_dataset(components=components)

    # Organize dataset
    downloader.organize_dataset()


if __name__ == '__main__':
    main()
