"""
Universal Data Loader for Multiple SLAM Datasets

Supports: KITTI, TUM RGB-D, ICL-NUIM, EuRoC, Newer College
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
from PIL import Image
import struct


class UniversalDataLoader:
    """
    Universal data loader that works with multiple SLAM datasets
    """

    def __init__(self, dataset_name: str, data_root: str, sequence: Optional[str] = None):
        """
        Initialize data loader

        Args:
            dataset_name: Name of dataset (kitti, tum_rgbd, icl_nuim, euroc, newer_college)
            data_root: Root directory of dataset
            sequence: Sequence identifier (dataset-specific)
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = Path(data_root)
        self.sequence = sequence

        # Initialize dataset-specific loader
        if self.dataset_name == 'kitti':
            self._init_kitti()
        elif self.dataset_name == 'tum_rgbd':
            self._init_tum_rgbd()
        elif self.dataset_name == 'icl_nuim':
            self._init_icl_nuim()
        elif self.dataset_name == 'euroc':
            self._init_euroc()
        elif self.dataset_name == 'newer_college':
            self._init_newer_college()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _init_kitti(self):
        """Initialize KITTI dataset"""
        self.sequence_dir = self.data_root / 'sequences' / self.sequence
        self.velodyne_dir = self.sequence_dir / 'velodyne'

        if not self.velodyne_dir.exists():
            raise ValueError(f"KITTI velodyne directory not found: {self.velodyne_dir}")

        self.data_files = sorted(self.velodyne_dir.glob('*.bin'))
        self.has_depth = True
        self.has_rgb = False

    def _init_tum_rgbd(self):
        """Initialize TUM RGB-D dataset"""
        # Find the first sequence directory if not specified
        if self.sequence is None:
            seq_dirs = list(self.data_root.glob('rgbd_dataset_*'))
            if not seq_dirs:
                raise ValueError(f"No TUM RGB-D sequences found in {self.data_root}")
        else:
            self.sequence_dir = self.data_root / self.sequence

        self.rgb_dir = self.sequence_dir / 'rgb'
        self.depth_dir = self.sequence_dir / 'depth'

        if not self.rgb_dir.exists():
            raise ValueError(f"TUM RGB directory not found: {self.rgb_dir}")

        self.rgb_files = sorted(self.rgb_dir.glob('*.png'))
        self.depth_files = sorted(self.depth_dir.glob('*.png')) if self.depth_dir.exists() else []
        self.data_files = self.rgb_files
        self.has_depth = True
        self.has_rgb = True

    def _init_icl_nuim(self):
        """Initialize ICL-NUIM dataset"""
        # Find the first sequence directory if not specified
        if self.sequence is None:
            seq_dirs = list(self.data_root.glob('*_traj*_frei_png'))
            if not seq_dirs:
                raise ValueError(f"No ICL-NUIM sequences found in {self.data_root}")
            self.sequence_dir = seq_dirs[0]
        else:
            self.sequence_dir = self.data_root / self.sequence

        if not self.sequence_dir.exists():
            raise ValueError(f"ICL-NUIM sequence not found: {self.sequence_dir}")

        # ICL-NUIM has scene*.png for RGB and scene*.depth for depth
        self.rgb_files = sorted(self.sequence_dir.glob('scene_*.png'))
        self.depth_files = sorted(self.sequence_dir.glob('*.depth'))

        # Use RGB files as primary data files
        self.data_files = self.rgb_files if self.rgb_files else self.depth_files
        self.has_depth = len(self.depth_files) > 0
        self.has_rgb = len(self.rgb_files) > 0

        print(f"ICL-NUIM: Found {len(self.rgb_files)} RGB images and {len(self.depth_files)} depth images")

    def _init_euroc(self):
        """Initialize EuRoC dataset"""
        if self.sequence is None:
            seq_dirs = list(self.data_root.glob('*_0*'))
            if not seq_dirs:
                raise ValueError(f"No EuRoC sequences found in {self.data_root}")
            self.sequence_dir = seq_dirs[0]
        else:
            self.sequence_dir = self.data_root / self.sequence

        cam0_dir = self.sequence_dir / 'mav0' / 'cam0' / 'data'
        if not cam0_dir.exists():
            raise ValueError(f"EuRoC camera directory not found: {cam0_dir}")

        self.data_files = sorted(cam0_dir.glob('*.png'))
        self.has_depth = False
        self.has_rgb = True

    def _init_newer_college(self):
        """Initialize Newer College dataset"""
        if self.sequence is None:
            seq_dirs = list(self.data_root.glob('0*_*_experiment'))
            if not seq_dirs:
                raise ValueError(f"No Newer College sequences found in {self.data_root}")
            self.sequence_dir = seq_dirs[0]
        else:
            self.sequence_dir = self.data_root / self.sequence

        lidar_dir = self.sequence_dir / 'ouster_scan'
        if not lidar_dir.exists():
            raise ValueError(f"Newer College lidar directory not found: {lidar_dir}")

        self.data_files = sorted(lidar_dir.glob('*.bin'))
        self.has_depth = True
        self.has_rgb = False

    def __len__(self) -> int:
        """Return number of frames/scans"""
        return len(self.data_files)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Load data at index and convert to point cloud

        Returns:
            Point cloud as numpy array (N, 3) or (N, 4) with intensity
        """
        if self.dataset_name == 'kitti':
            return self._load_kitti_scan(idx)
        elif self.dataset_name == 'tum_rgbd':
            return self._load_tum_rgbd_scan(idx)
        elif self.dataset_name == 'icl_nuim':
            return self._load_icl_nuim_scan(idx)
        elif self.dataset_name == 'euroc':
            return self._load_euroc_scan(idx)
        elif self.dataset_name == 'newer_college':
            return self._load_newer_college_scan(idx)

    def _load_kitti_scan(self, idx: int) -> np.ndarray:
        """Load KITTI point cloud"""
        points = np.fromfile(self.data_files[idx], dtype=np.float32).reshape(-1, 4)
        return points

    def _load_tum_rgbd_scan(self, idx: int) -> np.ndarray:
        """Load TUM RGB-D and convert to point cloud"""
        # Load RGB and depth
        rgb = np.array(Image.open(self.rgb_files[idx]))

        if idx < len(self.depth_files):
            depth = np.array(Image.open(self.depth_files[idx])).astype(np.float32) / 5000.0  # Scale factor
        else:
            # No depth available
            return np.zeros((0, 4))

        # Convert to point cloud (simple unprojection)
        points = self._depth_to_pointcloud(depth, rgb)
        return points

    def _load_icl_nuim_scan(self, idx: int) -> np.ndarray:
        """Load ICL-NUIM and convert to point cloud"""
        # Load RGB
        if idx < len(self.rgb_files):
            rgb = np.array(Image.open(self.rgb_files[idx]))
        else:
            rgb = None

        # Load depth - ICL-NUIM uses .depth files (binary format)
        if idx < len(self.depth_files):
            depth = self._load_icl_depth(self.depth_files[idx])
        else:
            return np.zeros((0, 4))

        # Convert to point cloud
        points = self._depth_to_pointcloud(depth, rgb)
        return points

    def _load_icl_depth(self, depth_file: Path) -> np.ndarray:
        """Load ICL-NUIM .depth file"""
        # ICL-NUIM depth files are binary float32
        with open(depth_file, 'rb') as f:
            # Read dimensions if stored, otherwise assume 640x480
            depth_data = np.fromfile(f, dtype=np.float32)

        # Try to reshape to standard ICL-NUIM size
        if len(depth_data) == 640 * 480:
            depth = depth_data.reshape(480, 640)
        elif len(depth_data) == 307200:  # Another common size
            depth = depth_data.reshape(480, 640)
        else:
            # Try square root for square images
            side = int(np.sqrt(len(depth_data)))
            if side * side == len(depth_data):
                depth = depth_data.reshape(side, side)
            else:
                # Default to 640x480 and truncate/pad
                depth = np.zeros((480, 640), dtype=np.float32)
                depth.flat[:len(depth_data)] = depth_data[:480*640]

        return depth

    def _load_euroc_scan(self, idx: int) -> np.ndarray:
        """Load EuRoC camera image (stereo, no depth)"""
        # EuRoC doesn't have depth, so we return empty for now
        # In a real application, you'd use stereo matching
        return np.zeros((0, 4))

    def _load_newer_college_scan(self, idx: int) -> np.ndarray:
        """Load Newer College lidar scan"""
        # Similar format to KITTI
        points = np.fromfile(self.data_files[idx], dtype=np.float32).reshape(-1, 4)
        return points

    def _depth_to_pointcloud(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None,
                            fx: float = 525.0, fy: float = 525.0,
                            cx: float = 319.5, cy: float = 239.5) -> np.ndarray:
        """
        Convert depth image to point cloud

        Args:
            depth: Depth image (H, W)
            rgb: RGB image (H, W, 3) optional
            fx, fy, cx, cy: Camera intrinsics

        Returns:
            Point cloud (N, 3) or (N, 4) with intensity
        """
        h, w = depth.shape

        # Create coordinate grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter out invalid depths
        valid = z > 0
        x = x[valid]
        y = y[valid]
        z = z[valid]

        points_xyz = np.stack([x, y, z], axis=-1)

        # Add intensity/color if RGB is available
        if rgb is not None:
            # Use grayscale as intensity
            intensity = np.mean(rgb[valid], axis=-1, keepdims=True) / 255.0
            points = np.hstack([points_xyz, intensity])
        else:
            # Add dummy intensity
            intensity = np.ones((len(points_xyz), 1)) * 0.5
            points = np.hstack([points_xyz, intensity])

        return points

    def get_sequence_name(self) -> str:
        """Get current sequence name"""
        if hasattr(self, 'sequence_dir'):
            return self.sequence_dir.name
        return str(self.sequence)
