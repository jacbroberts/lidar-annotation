"""
Point Cloud Processing Utilities

Utilities for loading, preprocessing, and manipulating KITTI point cloud data.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Tuple, Optional, List
import torch


class PointCloudProcessor:
    """Handles point cloud loading and preprocessing for KITTI dataset"""

    @staticmethod
    def load_kitti_bin(bin_path: str) -> np.ndarray:
        """
        Load KITTI point cloud from .bin file

        Args:
            bin_path: Path to .bin file

        Returns:
            numpy array of shape (N, 4) with columns [x, y, z, intensity]
        """
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    @staticmethod
    def filter_points(points: np.ndarray,
                      x_range: Tuple[float, float] = (-50, 50),
                      y_range: Tuple[float, float] = (-50, 50),
                      z_range: Tuple[float, float] = (-3, 10)) -> np.ndarray:
        """
        Filter points within specified range

        Args:
            points: Point cloud array (N, 4)
            x_range: (min, max) range for x coordinate
            y_range: (min, max) range for y coordinate
            z_range: (min, max) range for z coordinate

        Returns:
            Filtered point cloud
        """
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        )
        return points[mask]

    @staticmethod
    def voxel_downsample(points: np.ndarray,
                         voxel_size: float = 0.1) -> np.ndarray:
        """
        Downsample point cloud using voxel grid

        Args:
            points: Point cloud array (N, 4)
            voxel_size: Size of voxel grid

        Returns:
            Downsampled point cloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downsampled.points)

        # Preserve intensity through nearest neighbor
        if points.shape[1] > 3:
            from scipy.spatial import cKDTree
            tree = cKDTree(points[:, :3])
            _, indices = tree.query(downsampled_points)
            intensity = points[indices, 3:]
            downsampled_points = np.hstack([downsampled_points, intensity])

        return downsampled_points

    @staticmethod
    def compute_normals(points: np.ndarray,
                        search_radius: float = 0.5,
                        max_nn: int = 30) -> np.ndarray:
        """
        Compute normals for point cloud

        Args:
            points: Point cloud array (N, 3 or 4)
            search_radius: Radius for normal estimation
            max_nn: Maximum number of neighbors

        Returns:
            Normal vectors (N, 3)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius,
                max_nn=max_nn
            )
        )

        return np.asarray(pcd.normals)

    @staticmethod
    def to_tensor(points: np.ndarray,
                  device: str = 'cuda') -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor

        Args:
            points: Point cloud array
            device: Target device ('cuda' or 'cpu')

        Returns:
            PyTorch tensor
        """
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU")

        return torch.from_numpy(points).float().to(device)

    @staticmethod
    def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize point cloud to zero mean and unit scale

        Args:
            points: Point cloud array (N, 3 or more)

        Returns:
            Normalized points and normalization parameters
        """
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))

        xyz_normalized = xyz_centered / (max_dist + 1e-8)

        normalized_points = points.copy()
        normalized_points[:, :3] = xyz_normalized

        params = {
            'centroid': centroid,
            'scale': max_dist
        }

        return normalized_points, params

    @staticmethod
    def denormalize_points(points: np.ndarray, params: dict) -> np.ndarray:
        """
        Denormalize points back to original scale

        Args:
            points: Normalized point cloud
            params: Normalization parameters from normalize_points

        Returns:
            Denormalized points
        """
        denormalized = points.copy()
        denormalized[:, :3] = points[:, :3] * params['scale'] + params['centroid']
        return denormalized

    @staticmethod
    def extract_features(points: np.ndarray) -> np.ndarray:
        """
        Extract geometric features from point cloud

        Args:
            points: Point cloud array (N, 3+)

        Returns:
            Feature array (N, F)
        """
        xyz = points[:, :3]

        # Height feature
        height = xyz[:, 2:3]

        # Distance from origin
        distance = np.linalg.norm(xyz, axis=1, keepdims=True)

        # Intensity (if available)
        if points.shape[1] > 3:
            intensity = points[:, 3:4]
            features = np.hstack([xyz, height, distance, intensity])
        else:
            features = np.hstack([xyz, height, distance])

        return features

    @staticmethod
    def create_batches(points: np.ndarray,
                       batch_size: int = 4096) -> List[np.ndarray]:
        """
        Split point cloud into batches for processing

        Args:
            points: Point cloud array
            batch_size: Number of points per batch

        Returns:
            List of point cloud batches
        """
        n_points = len(points)
        n_batches = (n_points + batch_size - 1) // batch_size

        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            batches.append(points[start_idx:end_idx])

        return batches


class KITTIDataLoader:
    """Data loader for KITTI Odometry dataset"""

    def __init__(self, data_root: str, sequence: str = '00'):
        """
        Initialize data loader

        Args:
            data_root: Root directory of KITTI dataset
            sequence: Sequence number (e.g., '00', '01', ...)
        """
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.sequence_dir = self.data_root / 'sequences' / sequence
        self.velodyne_dir = self.sequence_dir / 'velodyne'

        if not self.velodyne_dir.exists():
            raise ValueError(f"Velodyne directory not found: {self.velodyne_dir}")

        self.scan_files = sorted(self.velodyne_dir.glob('*.bin'))
        self.processor = PointCloudProcessor()

    def __len__(self) -> int:
        """Return number of scans in sequence"""
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Load point cloud at index"""
        return self.processor.load_kitti_bin(str(self.scan_files[idx]))

    def get_scan_path(self, idx: int) -> Path:
        """Get path to scan file"""
        return self.scan_files[idx]

    def load_calibration(self) -> dict:
        """Load calibration data for the sequence"""
        calib_file = self.sequence_dir / 'calib.txt'

        if not calib_file.exists():
            return {}

        calib_data = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib_data[key.strip()] = np.fromstring(
                        value.strip(),
                        sep=' '
                    )

        return calib_data

    def load_poses(self) -> Optional[np.ndarray]:
        """Load ground truth poses if available"""
        poses_file = self.data_root / 'poses' / f'{self.sequence}.txt'

        if not poses_file.exists():
            return None

        poses = np.loadtxt(poses_file)
        # Reshape to 4x4 transformation matrices
        poses = poses.reshape(-1, 3, 4)

        return poses
