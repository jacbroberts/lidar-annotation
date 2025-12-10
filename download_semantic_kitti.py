"""
Download Semantic KITTI Labels

Semantic KITTI adds semantic labels to KITTI Odometry sequences 00-10.
This script downloads the label files needed for training SalsaNext.

Website: http://www.semantic-kitti.org/
"""

import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import argparse


def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress bar"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Download Semantic KITTI labels')
    parser.add_argument('--data_root', type=str, default='./data/kitti',
                       help='Path to KITTI dataset root')
    parser.add_argument('--keep_zip', action='store_true',
                       help='Keep zip file after extraction')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Semantic KITTI Label Downloader")
    print("=" * 60)

    # Semantic KITTI label data URL
    # Note: You may need to register on semantic-kitti.org to get access
    label_url = "http://semantic-kitti.org/assets/data_odometry_labels.zip"

    print("\nNote: Semantic KITTI requires registration for download access.")
    print("If the automatic download fails:")
    print("  1. Visit: http://www.semantic-kitti.org/dataset.html")
    print("  2. Register and download 'data_odometry_labels.zip' (~800MB)")
    print(f"  3. Place it in: {data_root}")
    print(f"  4. Run this script again with --keep_zip")
    print()

    # Check if zip already exists
    zip_path = data_root / "data_odometry_labels.zip"

    if not zip_path.exists():
        print(f"Downloading Semantic KITTI labels (~800MB)...")
        print(f"URL: {label_url}")
        print()

        try:
            download_file(label_url, zip_path)
            print("✓ Download complete!")
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            print("\nPlease download manually:")
            print(f"  1. Visit: http://www.semantic-kitti.org/dataset.html")
            print(f"  2. Download 'data_odometry_labels.zip'")
            print(f"  3. Place in: {data_root}")
            print(f"  4. Run: python {__file__} --keep_zip")
            return
    else:
        print(f"Found existing zip file: {zip_path}")

    # Extract
    print()
    extract_zip(zip_path, data_root)

    # Verify extraction
    sequences_dir = data_root / 'sequences'
    if not sequences_dir.exists():
        print(f"\n✗ Extraction failed - sequences directory not found")
        return

    # Check for label directories
    label_count = 0
    for seq_dir in sequences_dir.iterdir():
        if seq_dir.is_dir():
            labels_dir = seq_dir / 'labels'
            if labels_dir.exists():
                num_labels = len(list(labels_dir.glob('*.label')))
                label_count += num_labels
                print(f"  ✓ Sequence {seq_dir.name}: {num_labels} labels")

    print()
    if label_count > 0:
        print(f"✓ Successfully extracted {label_count} label files!")
        print()
        print("You can now train SalsaNext:")
        print("  python train_salsanext.py")
        print()
        print("Or for a quick test:")
        print("  python train_salsanext.py --quick_test")
    else:
        print("✗ No labels found after extraction")
        print("Please check the extraction manually")

    # Cleanup
    if not args.keep_zip:
        print()
        print(f"Removing zip file: {zip_path}")
        zip_path.unlink()

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
