"""
SalsaNext Inference Script

This script loads a trained SalsaNext model and performs semantic segmentation
on LiDAR point clouds. Use this to integrate SalsaNext into your pipeline:

LiDAR scan -> SalsaNext (this script) -> Segmented point cloud -> NLP model

Usage:
    python salsanext_inference.py --checkpoint path/to/salsanext_best.pth --input scan.bin
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from models.salsanext.SalsaNext import SalsaNext


class SalsaNextSegmenter:
    """Wrapper for SalsaNext inference"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize SalsaNext model for inference

        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Range image parameters (same as training)
        self.proj_H = 64
        self.proj_W = 2048
        self.proj_fov_up = 3.0
        self.proj_fov_down = -25.0

        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = SalsaNext(nclasses=20).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def _point_cloud_to_range_image(self, points):
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
        point_to_pixel = np.zeros(len(points), dtype=np.int32)  # Map point index to pixel

        # Fill range image (keep closest point per pixel)
        order = np.argsort(depth)[::-1]  # Far to near
        for idx in order:
            px, py = proj_x[idx], proj_y[idx]
            range_image[py, px, :3] = xyz[idx]
            range_image[py, px, 3] = depth[idx]
            range_image[py, px, 4] = remission[idx]
            point_to_pixel[idx] = py * self.proj_W + px

        # Transpose to (C, H, W) for PyTorch
        range_image = range_image.transpose(2, 0, 1)

        return range_image, proj_x, proj_y, point_to_pixel

    def segment(self, points):
        """
        Perform semantic segmentation on point cloud

        Args:
            points: numpy array of shape (N, 4) containing [x, y, z, remission]

        Returns:
            labels: numpy array of shape (N,) containing semantic labels (0-19)
        """
        # Project to range image
        range_image, proj_x, proj_y, point_to_pixel = self._point_cloud_to_range_image(points)

        # Convert to tensor and add batch dimension
        range_tensor = torch.from_numpy(range_image).unsqueeze(0).float().to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(range_tensor)  # (1, 20, H, W)
            predictions = output.argmax(dim=1).squeeze(0)  # (H, W)

        # Map predictions back to points
        predictions_np = predictions.cpu().numpy()
        labels = np.zeros(len(points), dtype=np.int32)

        for i in range(len(points)):
            py = proj_y[i]
            px = proj_x[i]
            labels[i] = predictions_np[py, px]

        return labels

    def segment_file(self, scan_path):
        """
        Load and segment a .bin point cloud file

        Args:
            scan_path: Path to .bin file (KITTI format)

        Returns:
            points: numpy array (N, 4)
            labels: numpy array (N,)
        """
        # Load point cloud
        points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

        # Segment
        labels = self.segment(points)

        return points, labels


def main():
    parser = argparse.ArgumentParser(description='SalsaNext Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained SalsaNext checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input point cloud (.bin file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predicted labels (.label file)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Initialize segmenter
    segmenter = SalsaNextSegmenter(args.checkpoint, device=args.device)

    # Run inference
    print(f"\nProcessing: {args.input}")
    points, labels = segmenter.segment_file(args.input)

    print(f"Point cloud size: {len(points)} points")
    print(f"Unique labels: {np.unique(labels)}")

    # Class distribution
    print("\nLabel distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label:2d}: {count:6d} points ({100*count/len(labels):.1f}%)")

    # Save labels if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in Semantic KITTI format (uint32)
        labels_uint32 = labels.astype(np.uint32)
        labels_uint32.tofile(output_path)
        print(f"\nLabels saved to: {output_path}")

    print("\nSegmentation complete!")


if __name__ == '__main__':
    main()
