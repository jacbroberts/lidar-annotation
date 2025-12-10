"""
Base class for dataset downloaders with common functionality
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Any


class BaseDownloader:
    """Base class for all dataset downloaders"""

    def __init__(self, root_dir: str, config: Dict[str, Any], download_options: Dict[str, Any]):
        """
        Initialize the downloader

        Args:
            root_dir: Root directory to store the dataset
            config: Dataset-specific configuration
            download_options: Download options (chunk size, retries, etc.)
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root_dir / 'raw'
        self.raw_dir.mkdir(exist_ok=True)

        self.config = config
        self.download_options = download_options

        # Create optimized session
        self.session = self._create_session()

    def _create_session(self):
        """Create a requests session with retry logic and connection pooling"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.download_options.get('max_retries', 10),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set timeout defaults
        timeout_config = self.download_options.get('timeout', {})
        session.timeout = (
            timeout_config.get('connect', 10),
            timeout_config.get('read', 300)
        )

        return session

    def download_file(self, url: str, dest_path: Path, chunk_size: int = None):
        """
        Download a file with progress bar, resume capability, and retry logic

        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Size of chunks to download
        """
        if chunk_size is None:
            chunk_size = self.download_options.get('chunk_size', 1024*1024)

        # Check if file exists
        resume_byte_pos = 0
        temp_path = Path(str(dest_path) + '.partial')

        if dest_path.exists():
            print(f"File already exists: {dest_path}")
            return

        if temp_path.exists() and self.download_options.get('resume_downloads', True):
            resume_byte_pos = temp_path.stat().st_size
            print(f"Resuming download from byte {resume_byte_pos}")

        # Prepare headers for resume
        headers = {}
        if resume_byte_pos > 0:
            headers['Range'] = f'bytes={resume_byte_pos}-'

        print(f"Downloading: {url}")

        max_retries = self.download_options.get('max_retries', 5)
        retry_delay = self.download_options.get('retry_delay', 5)

        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=(
                        self.download_options.get('timeout', {}).get('connect', 10),
                        self.download_options.get('timeout', {}).get('read', 300)
                    )
                )
                response.raise_for_status()

                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                if resume_byte_pos > 0:
                    total_size += resume_byte_pos

                # Open file in append mode if resuming
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
                        if data:
                            size = f.write(data)
                            pbar.update(size)

                # Download successful, rename temp file
                temp_path.rename(dest_path)
                print(f"Download complete: {dest_path}")
                return

            except (requests.exceptions.RequestException, IOError) as e:
                if attempt < max_retries - 1:
                    print(f"\nDownload error: {e}")
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(retry_delay)
                    # Update resume position
                    if temp_path.exists():
                        resume_byte_pos = temp_path.stat().st_size
                        headers['Range'] = f'bytes={resume_byte_pos}-'
                else:
                    print(f"\nFailed to download after {max_retries} attempts")
                    raise

    def extract_archive(self, archive_path: Path, extract_to: Path):
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
        elif archive_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path}")

        print(f"Extraction complete: {archive_path}")

    def download_dataset(self):
        """Download the dataset - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement download_dataset()")

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
