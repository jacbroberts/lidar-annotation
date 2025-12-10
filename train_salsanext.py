"""
SalsaNext Training Script for KITTI Semantic Segmentation

This script trains the SalsaNext model on Semantic KITTI dataset.
For quick pipeline testing, you can use --quick_test flag.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse

from models.salsanext.SalsaNext import SalsaNext


class SemanticKITTIDataset(Dataset):
    """
    Dataset loader for Semantic KITTI

    Semantic KITTI adds semantic labels to KITTI Odometry sequences 00-10.
    Download from: http://www.semantic-kitti.org/
    """

    def __init__(self, root_dir, sequences, config):
        """
        Args:
            root_dir: Path to KITTI dataset root
            sequences: List of sequence numbers to use (e.g., ['00', '01'])
            config: Training configuration dict
        """
        self.root_dir = Path(root_dir)
        self.sequences = sequences
        self.config = config

        # SalsaNext range image parameters
        self.proj_H = 64  # Range image height
        self.proj_W = 2048  # Range image width
        self.proj_fov_up = 3.0
        self.proj_fov_down = -25.0

        # Collect all scan files
        self.scan_files = []
        self.label_files = []

        for seq in sequences:
            seq_path = self.root_dir / 'sequences' / seq
            velodyne_path = seq_path / 'velodyne'
            labels_path = seq_path / 'labels'  # Semantic KITTI labels

            if not labels_path.exists():
                print(f"WARNING: No labels found for sequence {seq}")
                print(f"  Expected: {labels_path}")
                print(f"  You need to download Semantic KITTI labels from:")
                print(f"  http://www.semantic-kitti.org/dataset.html")
                continue

            # Get all scan files
            scan_files = sorted(velodyne_path.glob('*.bin'))
            for scan_file in scan_files:
                label_file = labels_path / (scan_file.stem + '.label')
                if label_file.exists():
                    self.scan_files.append(scan_file)
                    self.label_files.append(label_file)

        print(f"Loaded {len(self.scan_files)} scans from sequences: {sequences}")

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        # Load point cloud
        points = np.fromfile(self.scan_files[idx], dtype=np.float32).reshape(-1, 4)

        # Load labels
        labels = np.fromfile(self.label_files[idx], dtype=np.uint32)
        labels = labels & 0xFFFF  # Extract semantic label (lower 16 bits)

        # Remap labels to learning map (0-19)
        labels = self._remap_labels(labels)

        # Project to range image
        range_image, label_image, mask = self._point_cloud_to_range_image(points, labels)

        # Convert to tensors
        range_image = torch.from_numpy(range_image).float()
        label_image = torch.from_numpy(label_image).long()
        mask = torch.from_numpy(mask).float()

        return range_image, label_image, mask

    def _remap_labels(self, labels):
        """Remap Semantic KITTI labels to learning map (0-19)"""
        # Semantic KITTI learning map
        learning_map = {
            0: 0,     # "unlabeled"
            1: 0,     # "outlier" -> unlabeled
            10: 1,    # "car"
            11: 2,    # "bicycle"
            13: 5,    # "bus" -> other-vehicle
            15: 3,    # "motorcycle"
            16: 5,    # "on-rails" -> other-vehicle
            18: 4,    # "truck"
            20: 5,    # "other-vehicle"
            30: 6,    # "person"
            31: 7,    # "bicyclist"
            32: 8,    # "motorcyclist"
            40: 9,    # "road"
            44: 10,   # "parking"
            48: 11,   # "sidewalk"
            49: 12,   # "other-ground"
            50: 13,   # "building"
            51: 14,   # "fence"
            52: 0,    # "other-structure" -> unlabeled
            60: 9,    # "lane-marking" -> road
            70: 15,   # "vegetation"
            71: 16,   # "trunk"
            72: 17,   # "terrain"
            80: 18,   # "pole"
            81: 19,   # "traffic-sign"
            99: 0,    # "other-object" -> unlabeled
            252: 1,   # "moving-car" -> car
            253: 7,   # "moving-bicyclist" -> bicyclist
            254: 6,   # "moving-person" -> person
            255: 8,   # "moving-motorcyclist" -> motorcyclist
            256: 5,   # "moving-on-rails" -> other-vehicle
            257: 5,   # "moving-bus" -> other-vehicle
            258: 4,   # "moving-truck" -> truck
            259: 5,   # "moving-other-vehicle" -> other-vehicle
        }

        # Remap
        remapped = np.zeros_like(labels)
        for original, mapped in learning_map.items():
            remapped[labels == original] = mapped

        return remapped

    def _point_cloud_to_range_image(self, points, labels):
        """Project 3D point cloud to 2D range image"""
        xyz = points[:, :3]
        remission = points[:, 3]

        # Calculate range
        depth = np.linalg.norm(xyz, axis=1)

        # Calculate angles
        yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
        pitch = np.arcsin(xyz[:, 2] / (depth + 1e-8))

        # Project to image coordinates
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        # Scale to image size
        proj_x = proj_x * self.proj_W
        proj_y = proj_y * self.proj_H

        # Round and clip
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)

        # Create range image (channels: x, y, z, depth, remission)
        range_image = np.zeros((self.proj_H, self.proj_W, 5), dtype=np.float32)
        label_image = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        mask = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)

        # Fill range image (keep closest point per pixel)
        order = np.argsort(depth)[::-1]  # Far to near (so near points overwrite)
        for idx in order:
            px, py = proj_x[idx], proj_y[idx]
            range_image[py, px, :3] = xyz[idx]
            range_image[py, px, 3] = depth[idx]
            range_image[py, px, 4] = remission[idx]
            label_image[py, px] = labels[idx]
            mask[py, px] = 1.0

        # Transpose to (C, H, W) for PyTorch
        range_image = range_image.transpose(2, 0, 1)

        return range_image, label_image, mask


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for range_images, labels, masks in pbar:
        range_images = range_images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(range_images)

        # Reshape for loss computation
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)
        masks = masks.view(-1)

        # Compute loss (only on valid pixels)
        valid_mask = masks > 0
        loss = criterion(outputs[valid_mask], labels[valid_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += ((pred == labels) & valid_mask).sum().item()
        total += valid_mask.sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                         'acc': f'{100.0 * correct / total:.2f}%'})

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for range_images, labels, masks in pbar:
            range_images = range_images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(range_images)

            # Reshape
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            masks = masks.view(-1)

            # Compute loss
            valid_mask = masks > 0
            loss = criterion(outputs[valid_mask], labels[valid_mask])

            # Statistics
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += ((pred == labels) & valid_mask).sum().item()
            total += valid_mask.sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                            'acc': f'{100.0 * correct / total:.2f}%'})

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train SalsaNext on Semantic KITTI')
    parser.add_argument('--data_root', type=str, default='./data/kitti',
                       help='Path to KITTI dataset root')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./models/weights',
                       help='Directory to save checkpoints')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode: 1 epoch, sequence 00 only')

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        print("=" * 60)
        print("QUICK TEST MODE ENABLED")
        print("  - Training on sequence 00 only")
        print("  - 1 epoch")
        print("  - This is just to verify the pipeline works!")
        print("=" * 60)
        args.epochs = 1
        train_sequences = ['00']
        val_sequences = ['00']
    else:
        # Standard train/val split for Semantic KITTI
        train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        val_sequences = ['08']

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("\nLoading training data...")
    train_dataset = SemanticKITTIDataset(
        args.data_root,
        train_sequences,
        config={}
    )

    print("\nLoading validation data...")
    val_dataset = SemanticKITTIDataset(
        args.data_root,
        val_sequences,
        config={}
    )

    if len(train_dataset) == 0:
        print("\nERROR: No training data found!")
        print("\nYou need Semantic KITTI labels. Two options:")
        print("\n1. DOWNLOAD SEMANTIC KITTI (Recommended for proper training):")
        print("   - Visit: http://www.semantic-kitti.org/dataset.html")
        print("   - Download: 'SemanticKITTI label data' (~800MB)")
        print("   - Extract to: data/kitti/sequences/XX/labels/")
        print("\n2. QUICK TEST MODE (Just to verify pipeline):")
        print("   - Use heuristic segmentation temporarily")
        print("   - Change config: model_type: 'heuristic'")
        print("   - Run: python demo.py")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print("\nInitializing SalsaNext model...")
    model = SalsaNext(nclasses=20).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = FocalLoss(gamma=2.0, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint_path = save_dir / f'salsanext_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / 'salsanext_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, best_path)
            print(f"✓ Best model saved: {best_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {save_dir / 'salsanext_best.pth'}")
    print("\nTo use in your pipeline, update default_config.yaml:")
    print("  segmentation:")
    print("    model_type: 'salsanext'")
    print(f"    checkpoint_path: '{save_dir / 'salsanext_best.pth'}'")
    print("=" * 60)


if __name__ == '__main__':
    main()
