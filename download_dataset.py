"""
Unified Dataset Download Script

This is a simple wrapper around the dataset manager for easy dataset downloading.
Configure your dataset in dataset_config.yaml and run this script.

Usage:
    # Download the active dataset
    python download_dataset.py

    # List all available datasets
    python download_dataset.py --list

    # Download a specific dataset
    python download_dataset.py --dataset tum_rgbd

    # Set active dataset and download
    python download_dataset.py --set-active icl_nuim --download
"""

import argparse
from archive.dataset_manager import DatasetManager


def main():
    parser = argparse.ArgumentParser(
        description='Unified SLAM Dataset Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python download_dataset.py --list

  # Download TUM RGB-D dataset
  python download_dataset.py --dataset tum_rgbd

  # Set ICL-NUIM as active and download
  python download_dataset.py --set-active icl_nuim --download

  # Download active dataset (from config)
  python download_dataset.py --download
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets with details'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify which dataset to download (kitti, tum_rgbd, icl_nuim, euroc, newer_college)'
    )

    parser.add_argument(
        '--set-active',
        type=str,
        dest='set_active',
        help='Set the active dataset in config'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download the dataset'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='dataset_config.yaml',
        help='Path to configuration file (default: dataset_config.yaml)'
    )

    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate a config file for the dataset (automatically done with --download)'
    )

    parser.add_argument(
        '--no-auto-config',
        action='store_true',
        help='Disable automatic config file generation when downloading'
    )

    args = parser.parse_args()

    # Initialize manager
    try:
        manager = DatasetManager(config_path=args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Please ensure dataset_config.yaml exists in the current directory.")
        return 1

    # Handle list command
    if args.list:
        manager.list_datasets()
        return 0

    # Handle set-active command or dataset switch
    if args.set_active:
        try:
            manager.set_active_dataset(args.set_active)
            manager.save_config()
            print(f"\nActive dataset changed to: {args.set_active}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Switch dataset if specified (for generate-config or download)
    if args.dataset:
        try:
            print(f"\nSwitching to dataset: {args.dataset}")
            manager.set_active_dataset(args.dataset)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Handle generate-config command
    if args.generate_config:
        try:
            manager.generate_dataset_config()
            return 0
        except Exception as e:
            print(f"Error generating config: {e}")
            return 1

    # Determine if we should auto-generate config
    auto_generate = not args.no_auto_config

    # Handle dataset-specific download (dataset already switched above if --dataset was specified)
    if args.dataset and not args.generate_config:
        try:
            manager.download_active_dataset(generate_config=auto_generate)
            return 0
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return 1

    # Handle download command (uses active dataset)
    if args.download:
        try:
            manager.download_active_dataset(generate_config=auto_generate)
            return 0
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return 1

    # No specific command - show help
    if not any([args.list, args.dataset, args.set_active, args.download]):
        print("Current configuration:")
        manager.list_datasets()
        print("\nUse --help to see available commands")
        print("\nQuick start:")
        print("  python download_dataset.py --list              # List datasets")
        print("  python download_dataset.py --dataset tum_rgbd  # Download TUM RGB-D")
        return 0


if __name__ == '__main__':
    exit(main())
