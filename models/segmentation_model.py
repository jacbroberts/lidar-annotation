"""
Segmentation Model Wrappers

Base classes and implementations for 3D point cloud segmentation models.
Supports integration with models like RandLA-Net, Cylinder3D, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from pathlib import Path


# KITTI Semantic Segmentation Class Names (Semantic KITTI)
KITTI_CLASS_NAMES = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}

# Color map for each class (RGB values normalized to 0-1, then converted to ANSI)
KITTI_CLASS_COLORS = {
    0: '\033[90m',      # unlabeled - gray
    1: '\033[94m',      # car - blue
    2: '\033[96m',      # bicycle - cyan
    3: '\033[36m',      # motorcycle - dark cyan
    4: '\033[34m',      # truck - dark blue
    5: '\033[35m',      # other-vehicle - magenta
    6: '\033[91m',      # person - bright red
    7: '\033[31m',      # bicyclist - red
    8: '\033[95m',      # motorcyclist - bright magenta
    9: '\033[90m',      # road - gray
    10: '\033[37m',     # parking - white
    11: '\033[93m',     # sidewalk - yellow
    12: '\033[33m',     # other-ground - dark yellow
    13: '\033[92m',     # building - bright green
    14: '\033[33m',     # fence - dark yellow
    15: '\033[32m',     # vegetation - green
    16: '\033[33m',     # trunk - brown (dark yellow)
    17: '\033[37m',     # terrain - light gray
    18: '\033[36m',     # pole - cyan
    19: '\033[93m',     # traffic-sign - bright yellow
}
ANSI_RESET = '\033[0m'


class BaseSegmentationModel(ABC):
    """Base class for segmentation models"""

    def __init__(self, num_classes: int = 20, device: str = 'cuda'):
        """
        Initialize segmentation model

        Args:
            num_classes: Number of semantic classes
            device: Device to run model on
        """
        self.num_classes = num_classes
        self.device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")

        self.model = None
        self.class_names = KITTI_CLASS_NAMES

    def get_class_name(self, label: int) -> str:
        """Get class name from label ID"""
        return self.class_names.get(label, f'class_{label}')

    def get_class_color(self, label: int) -> str:
        """Get ANSI color code for class label"""
        return KITTI_CLASS_COLORS.get(label, '\033[37m')  # Default to white

    def get_colored_class_name(self, label: int) -> str:
        """Get colored class name string"""
        color = self.get_class_color(label)
        name = self.get_class_name(label)
        return f"{color}{name} (class {label}){ANSI_RESET}"

    def get_class_distribution(self, labels: np.ndarray, colored: bool = True) -> Dict:
        """
        Get distribution of classes in labels

        Args:
            labels: Array of predicted labels
            colored: Whether to include color information

        Returns:
            Dictionary mapping label IDs to info dict with name, count, and color
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        distribution = {}
        for label, count in zip(unique_labels, counts):
            label_int = int(label)
            distribution[label_int] = {
                'name': self.get_class_name(label_int),
                'count': int(count),
                'color': self.get_class_color(label_int) if colored else '',
                'colored_name': self.get_colored_class_name(label_int) if colored else self.get_class_name(label_int)
            }
        return distribution

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load model weights"""
        pass

    @abstractmethod
    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess point cloud for model input"""
        pass

    @abstractmethod
    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Predict semantic labels for point cloud

        Args:
            points: Point cloud array (N, 3+)

        Returns:
            Semantic labels (N,)
        """
        pass

    def extract_objects(self,
                        points: np.ndarray,
                        labels: np.ndarray,
                        min_points: int = 50) -> List[Dict]:
        """
        Extract individual objects from segmented point cloud

        Args:
            points: Point cloud array (N, 3+)
            labels: Semantic labels (N,)
            min_points: Minimum points for an object

        Returns:
            List of object dictionaries with points, label, and metadata
        """
        from scipy.spatial import cKDTree
        from sklearn.cluster import DBSCAN

        objects = []
        unique_labels = np.unique(labels)

        # For each semantic class, cluster into individual instances
        for sem_label in unique_labels:
            if sem_label == 0:  # Skip unlabeled
                continue

            # Get points belonging to this class
            mask = labels == sem_label
            class_points = points[mask]
            class_indices = np.where(mask)[0]

            if len(class_points) < min_points:
                continue

            # Adaptive DBSCAN parameters based on class type
            class_name = self.get_class_name(int(sem_label))

            # Adjust clustering parameters based on object type
            if class_name in ['road', 'sidewalk', 'terrain', 'parking', 'other-ground']:
                # Ground classes: large eps to keep ground as few clusters
                eps = 3.0
                min_samples = 100
            elif class_name in ['building', 'vegetation']:
                # Large structures: more tolerant clustering
                eps = 2.0
                min_samples = 50
            elif class_name in ['car', 'truck', 'other-vehicle']:
                # Vehicles: moderate clustering
                eps = 1.0
                min_samples = 20
            elif class_name in ['person', 'bicyclist', 'motorcyclist']:
                # People: tighter clustering
                eps = 0.8
                min_samples = 10
            elif class_name in ['pole', 'trunk']:
                # Vertical structures: very tight
                eps = 0.5
                min_samples = 10
            else:
                # Default
                eps = 1.0
                min_samples = 15

            # Cluster into instances using DBSCAN with adaptive parameters
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points[:, :3])
            instance_labels = clustering.labels_

            # Extract each instance
            for instance_id in np.unique(instance_labels):
                if instance_id == -1:  # Skip noise
                    continue

                instance_mask = instance_labels == instance_id
                instance_points = class_points[instance_mask]

                if len(instance_points) < min_points:
                    continue

                # Compute bounding box
                bbox_min = np.min(instance_points[:, :3], axis=0)
                bbox_max = np.max(instance_points[:, :3], axis=0)
                centroid = np.mean(instance_points[:, :3], axis=0)

                obj = {
                    'points': instance_points,
                    'semantic_label': int(sem_label),
                    'semantic_class': self.get_class_name(int(sem_label)),
                    'instance_id': len(objects),
                    'centroid': centroid,
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max,
                    'num_points': len(instance_points),
                    'original_indices': class_indices[instance_mask]
                }

                objects.append(obj)

        # Merge overlapping objects to fix multi-class issues
        objects = self._merge_overlapping_objects(objects)

        return objects

    def _merge_overlapping_objects(self, objects: List[Dict],
                                   iou_threshold: float = 0.3,
                                   distance_threshold: float = 2.0) -> List[Dict]:
        """
        Merge objects that are likely the same physical object.
        Fixes the issue where a single car has multiple class labels.

        Args:
            objects: List of object dictionaries
            iou_threshold: IoU threshold for merging
            distance_threshold: Distance threshold (meters) for merging

        Returns:
            List of merged objects
        """
        if len(objects) == 0:
            return []

        def compute_iou_3d(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
            """Compute 3D Intersection over Union"""
            inter_min = np.maximum(bbox1_min, bbox2_min)
            inter_max = np.minimum(bbox1_max, bbox2_max)

            if np.any(inter_min >= inter_max):
                return 0.0

            inter_volume = np.prod(inter_max - inter_min)
            vol1 = np.prod(bbox1_max - bbox1_min)
            vol2 = np.prod(bbox2_max - bbox2_min)
            union_volume = vol1 + vol2 - inter_volume

            return inter_volume / (union_volume + 1e-8)

        # Sort by number of points (largest first)
        objects = sorted(objects, key=lambda x: x['num_points'], reverse=True)

        merged = []
        used = set()

        for i, obj1 in enumerate(objects):
            if i in used:
                continue

            # Start with this object
            merged_points = [obj1['points']]
            merged_classes = [obj1['semantic_label']]
            merged_counts = [obj1['num_points']]
            used.add(i)

            # Find objects to merge
            for j, obj2 in enumerate(objects):
                if j <= i or j in used:
                    continue

                # Check distance between centroids
                dist = np.linalg.norm(obj1['centroid'] - obj2['centroid'])

                # Check 3D IoU
                iou = compute_iou_3d(obj1['bbox_min'], obj1['bbox_max'],
                                    obj2['bbox_min'], obj2['bbox_max'])

                # Merge if close or overlapping
                if dist < distance_threshold or iou > iou_threshold:
                    merged_points.append(obj2['points'])
                    merged_classes.append(obj2['semantic_label'])
                    merged_counts.append(obj2['num_points'])
                    used.add(j)

            # Combine all points
            all_points = np.vstack(merged_points)

            # Majority vote for semantic class (weighted by point count)
            class_votes = {}
            for cls, count in zip(merged_classes, merged_counts):
                class_votes[cls] = class_votes.get(cls, 0) + count
            final_class = max(class_votes.items(), key=lambda x: x[1])[0]

            # Create merged object
            merged_obj = {
                'points': all_points,
                'semantic_label': int(final_class),
                'semantic_class': self.get_class_name(int(final_class)),
                'instance_id': len(merged),
                'centroid': all_points[:, :3].mean(axis=0),
                'bbox_min': all_points[:, :3].min(axis=0),
                'bbox_max': all_points[:, :3].max(axis=0),
                'num_points': len(all_points),
            }

            merged.append(merged_obj)

        return merged


class HeuristicSegmentationModel(BaseSegmentationModel):
    """
    Rule-based heuristic segmentation using geometric features

    This model uses geometric properties (height, intensity, local features)
    to classify points into semantic categories. Works without training.

    KITTI Semantic Classes (Semantic KITTI):
    0: unlabeled, 1: car, 2: bicycle, 3: motorcycle, 4: truck, 5: other-vehicle
    6: person, 7: bicyclist, 8: motorcyclist, 9: road, 10: parking, 11: sidewalk
    12: other-ground, 13: building, 14: fence, 15: vegetation, 16: trunk, 17: terrain
    18: pole, 19: traffic-sign
    """

    def __init__(self, num_classes: int = 20, device: str = 'cuda'):
        super().__init__(num_classes, device)
        print("Using heuristic segmentation (no training required)")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """No model to load for heuristic approach"""
        pass

    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """No preprocessing needed, return as-is"""
        return points

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Predict semantic labels using geometric heuristics

        Args:
            points: Point cloud array (N, 4) with [x, y, z, intensity]

        Returns:
            Semantic labels (N,)
        """
        n_points = len(points)
        labels = np.zeros(n_points, dtype=np.int32)

        xyz = points[:, :3]
        z = xyz[:, 2]

        # Get intensity if available
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros(n_points)

        # Compute relative height from ground
        ground_height = self._estimate_ground_height(xyz)
        height_above_ground = z - ground_height

        # Compute local geometric features
        local_features = self._compute_local_features(xyz)

        # Rule-based classification
        # 1. Ground plane (road, parking, sidewalk, terrain)
        ground_mask = (z < ground_height + 0.2) & (z > ground_height - 0.5)

        # Road: low points with high intensity (reflective)
        road_mask = ground_mask & (intensity > np.percentile(intensity, 60))
        labels[road_mask] = 9  # road

        # Sidewalk: slightly elevated, lower intensity
        sidewalk_mask = ground_mask & ~road_mask & (height_above_ground > -0.1) & (height_above_ground < 0.3)
        labels[sidewalk_mask] = 11  # sidewalk

        # Terrain/other ground: rest of ground points
        terrain_mask = ground_mask & (labels == 0)
        labels[terrain_mask] = 17  # terrain

        # 2. Vegetation (trees, bushes): medium height, irregular, low intensity
        veg_mask = (
            (height_above_ground > 0.3) &
            (height_above_ground < 15.0) &
            (intensity < np.percentile(intensity, 40)) &
            (local_features['verticality'] < 0.6) &
            (labels == 0)
        )

        # Distinguish trunk from foliage
        trunk_mask = veg_mask & (local_features['verticality'] > 0.4) & (height_above_ground < 4.0)
        labels[trunk_mask] = 16  # trunk
        labels[veg_mask & ~trunk_mask] = 15  # vegetation

        # 3. Buildings: tall, vertical, large planar surfaces
        building_mask = (
            (height_above_ground > 2.0) &
            (local_features['verticality'] > 0.7) &
            (local_features['planarity'] > 0.6) &
            (labels == 0)
        )
        labels[building_mask] = 13  # building

        # 4. Poles and traffic signs: thin vertical structures
        pole_mask = (
            (height_above_ground > 1.0) &
            (height_above_ground < 10.0) &
            (local_features['verticality'] > 0.85) &
            (local_features['linearity'] > 0.7) &
            (labels == 0)
        )
        labels[pole_mask] = 18  # pole

        # Traffic signs: elevated small objects near poles
        sign_mask = (
            (height_above_ground > 2.0) &
            (height_above_ground < 6.0) &
            (local_features['planarity'] > 0.5) &
            (~pole_mask) &
            (labels == 0)
        )
        labels[sign_mask] = 19  # traffic-sign

        # 5. Vehicles: medium height, compact, moderate intensity
        vehicle_height_mask = (height_above_ground > 0.4) & (height_above_ground < 3.0)
        vehicle_mask = (
            vehicle_height_mask &
            (intensity > np.percentile(intensity, 30)) &
            (local_features['planarity'] > 0.4) &
            (labels == 0)
        )

        # Classify vehicle types based on height and size
        car_mask = vehicle_mask & (height_above_ground < 2.2)
        labels[car_mask] = 1  # car

        truck_mask = vehicle_mask & (height_above_ground >= 2.2)
        labels[truck_mask] = 4  # truck

        # 6. Persons: narrow vertical structures at person height
        person_mask = (
            (height_above_ground > 1.2) &
            (height_above_ground < 2.2) &
            (local_features['verticality'] > 0.6) &
            (local_features['linearity'] > 0.4) &
            (labels == 0)
        )
        labels[person_mask] = 6  # person

        # 7. Fence: low-medium height, linear structures
        fence_mask = (
            (height_above_ground > 0.5) &
            (height_above_ground < 2.5) &
            (local_features['linearity'] > 0.6) &
            (local_features['planarity'] > 0.3) &
            (labels == 0)
        )
        labels[fence_mask] = 14  # fence

        # Post-processing: Spatial label smoothing to fix inconsistencies
        labels = self._smooth_labels(xyz, labels, k=15)

        return labels

    def _smooth_labels(self, xyz: np.ndarray, labels: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Smooth labels using k-nearest neighbor voting to fix inconsistencies

        Args:
            xyz: Point coordinates (N, 3)
            labels: Initial label predictions (N,)
            k: Number of neighbors to consider

        Returns:
            Smoothed labels (N,)
        """
        from scipy.spatial import cKDTree
        from scipy.stats import mode

        # Build KD-tree
        tree = cKDTree(xyz)
        smoothed_labels = labels.copy()

        # Only smooth non-ground classes to preserve fine details
        non_ground_classes = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18, 19]

        # Sample points for efficiency
        n_points = len(xyz)
        sample_size = min(n_points, 5000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)

        for idx in sample_indices:
            current_label = labels[idx]

            # Only smooth if current label is in non-ground classes
            if current_label not in non_ground_classes:
                continue

            # Find k nearest neighbors
            _, neighbor_indices = tree.query(xyz[idx], k=min(k, n_points))
            neighbor_labels = labels[neighbor_indices]

            # Use majority voting
            mode_result = mode(neighbor_labels, keepdims=True)
            majority_label = mode_result.mode[0]
            majority_count = mode_result.count[0]

            # Only change if there's a strong majority (> 60%)
            if majority_count > 0.6 * k and majority_label != current_label:
                smoothed_labels[idx] = majority_label

        # Propagate smoothed labels to nearby points
        tree_sample = cKDTree(xyz[sample_indices])
        _, nearest_sample = tree_sample.query(xyz, k=1)

        # Only apply smoothing where it makes sense
        for i in range(n_points):
            if labels[i] in non_ground_classes:
                sample_idx = sample_indices[nearest_sample[i]]
                if smoothed_labels[sample_idx] != labels[sample_idx]:
                    # Check if nearby points agree
                    _, nearby = tree.query(xyz[i], k=5)
                    nearby_labels = labels[nearby]
                    if np.sum(nearby_labels == smoothed_labels[sample_idx]) >= 3:
                        smoothed_labels[i] = smoothed_labels[sample_idx]

        return smoothed_labels

    def _estimate_ground_height(self, xyz: np.ndarray, percentile: float = 5.0) -> float:
        """
        Estimate ground plane height

        Args:
            xyz: Point coordinates (N, 3)
            percentile: Percentile of z values to use as ground

        Returns:
            Estimated ground height
        """
        # Use low percentile of z values as ground estimate
        ground_height = np.percentile(xyz[:, 2], percentile)

        # Refine using RANSAC-like approach on low points
        low_points = xyz[xyz[:, 2] < ground_height + 0.5]
        if len(low_points) > 100:
            # Fit plane to lowest points
            from sklearn.linear_model import RANSACRegressor
            try:
                X = low_points[:, :2]  # x, y
                y = low_points[:, 2]   # z
                ransac = RANSACRegressor(random_state=0, min_samples=50)
                ransac.fit(X, y)
                # Use mean of inliers
                ground_height = np.mean(y[ransac.inlier_mask_])
            except:
                pass  # Fall back to percentile estimate

        return ground_height

    def _compute_local_features(self, xyz: np.ndarray, k: int = 20) -> dict:
        """
        Compute local geometric features using PCA on k-nearest neighbors

        Args:
            xyz: Point coordinates (N, 3)
            k: Number of neighbors for local analysis

        Returns:
            Dictionary with local features
        """
        from scipy.spatial import cKDTree

        n_points = len(xyz)
        features = {
            'verticality': np.zeros(n_points),
            'planarity': np.zeros(n_points),
            'linearity': np.zeros(n_points)
        }

        # Build KD-tree
        tree = cKDTree(xyz)

        # Sample points for efficiency (full computation would be slow)
        sample_indices = np.random.choice(n_points, min(n_points, 10000), replace=False)

        for idx in sample_indices:
            # Find k nearest neighbors
            _, indices = tree.query(xyz[idx], k=min(k, n_points))
            neighbors = xyz[indices]

            if len(neighbors) < 3:
                continue

            # Compute covariance matrix
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

            # Normalize eigenvalues
            eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)

            # Geometric features from eigenvalues
            e1, e2, e3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

            # Linearity: how 1D is the structure
            features['linearity'][idx] = (e1 - e2) / (e1 + 1e-8)

            # Planarity: how 2D is the structure
            features['planarity'][idx] = (e2 - e3) / (e1 + 1e-8)

            # Verticality: alignment with vertical axis
            normal = eigenvectors[:, np.argmin(eigenvalues)]
            features['verticality'][idx] = abs(normal[2])

        # Interpolate features to all points (simple nearest neighbor)
        tree_sample = cKDTree(xyz[sample_indices])
        for feature_name in features:
            _, nearest_idx = tree_sample.query(xyz, k=1)
            features[feature_name] = features[feature_name][sample_indices[nearest_idx]]

        return features


class SimpleSegmentationModel(BaseSegmentationModel):
    """
    Simple PointNet-style segmentation model for demonstration

    This is a lightweight model for testing the pipeline.
    Replace with RandLA-Net, Cylinder3D, etc. for production use.

    NOTE: This requires trained weights to work properly. Without training,
    use HeuristicSegmentationModel instead.
    """

    def __init__(self, num_classes: int = 20, device: str = 'cuda'):
        super().__init__(num_classes, device)
        # Initialize with 4 input channels to handle (x, y, z, intensity)
        self.model = SimplePointNet(num_classes=num_classes, input_dim=4)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load model weights"""
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: No checkpoint provided for SimpleSegmentationModel")
            print("This model uses randomly initialized weights and will NOT work properly")
            print("Consider using HeuristicSegmentationModel (model_type='heuristic') instead")

    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess point cloud for model input"""
        # Normalize coordinates
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
        xyz_normalized = xyz_centered / (max_dist + 1e-8)

        # Use xyz + features
        if points.shape[1] > 3:
            features = points[:, 3:]
            features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            processed = np.hstack([xyz_normalized, features_normalized])
        else:
            processed = xyz_normalized

        return torch.from_numpy(processed).float().to(self.device)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Predict semantic labels for point cloud"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(points)

            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)  # (1, N, F)

            # Forward pass
            logits = self.model(input_tensor)  # (1, N, num_classes)

            # Get predictions
            predictions = torch.argmax(logits, dim=2)  # (1, N)
            predictions = predictions.squeeze(0).cpu().numpy()

        return predictions


class SimplePointNet(nn.Module):
    """
    Simplified PointNet for point cloud segmentation

    This is a basic implementation for demonstration.
    For production, use models like RandLA-Net, Cylinder3D, etc.
    """

    def __init__(self, num_classes: int = 20, input_dim: int = 3):
        super().__init__()

        self.num_classes = num_classes

        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Segmentation head
        self.conv4 = nn.Conv1d(256 + 256, 256, 1)
        self.conv5 = nn.Conv1d(256, 128, 1)
        self.conv6 = nn.Conv1d(128, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (B, N, F) where B=batch, N=points, F=features

        Returns:
            Logits (B, N, num_classes)
        """
        batch_size, num_points, _ = x.shape

        # Transpose to (B, F, N) for conv1d
        x = x.transpose(1, 2)

        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Save 256-dim features for concatenation
        point_features = x

        # Global feature
        global_feature = torch.max(x, dim=2, keepdim=True)[0]
        global_feature = global_feature.repeat(1, 1, num_points)

        # Concatenate point and global features (256 + 256 = 512)
        x = torch.cat([point_features, global_feature], dim=1)

        # Segmentation head
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        # Transpose back to (B, N, num_classes)
        x = x.transpose(1, 2)

        return x


class SalsaNextWrapper(BaseSegmentationModel):
    """
    Wrapper for SalsaNext model (range-image based semantic segmentation)

    SalsaNext uses range images for efficient LiDAR segmentation.
    Paper: https://arxiv.org/abs/2003.03653
    Code: https://github.com/TiagoCortinhal/SalsaNext

    To use:
    1. Train SalsaNext or obtain pretrained weights
    2. Place checkpoint in models/weights/
    3. Provide checkpoint path when creating model
    """

    def __init__(self, num_classes: int = 20, device: str = 'cuda'):
        super().__init__(num_classes, device)
        self.proj_H = 64  # Height of range image
        self.proj_W = 2048  # Width of range image
        self.proj_fov_up = 3.0  # Field of view up (degrees)
        self.proj_fov_down = -25.0  # Field of view down (degrees)
        self.model = None

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load SalsaNext model from checkpoint

        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            print("WARNING: No valid checkpoint path provided for SalsaNext")
            print("Model will not work without trained weights.")
            print("Please provide a checkpoint trained on KITTI Semantic dataset.")
            return

        try:
            # Import SalsaNext architecture
            # You'll need to have the SalsaNext code available
            from models.salsanext.model import SalsaNext

            # Initialize model
            self.model = SalsaNext(nclasses=self.num_classes)
            self.model = self.model.to(self.device)

            # Load checkpoint
            print(f"Loading SalsaNext checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel training)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value
                             for key, value in state_dict.items()}

            self.model.load_state_dict(state_dict)

            self.model.eval()
            print("SalsaNext model loaded successfully!")

        except ImportError:
            print("ERROR: SalsaNext model code not found.")
            print("Please ensure SalsaNext implementation is in models/salsanext/")
            print("Clone from: https://github.com/TiagoCortinhal/SalsaNext")
            raise
        except Exception as e:
            print(f"ERROR loading SalsaNext model: {e}")
            raise

    def _point_cloud_to_range_image(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project point cloud to range image

        Args:
            points: Point cloud (N, 4) with [x, y, z, intensity]

        Returns:
            range_image: (H, W, 5) with [range, x, y, z, intensity]
            proj_mask: (H, W) valid pixel mask
        """
        # Extract coordinates and intensity
        xyz = points[:, :3]
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros(len(points))

        # Compute range
        depth = np.linalg.norm(xyz, axis=1)

        # Compute angles
        yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
        pitch = np.arcsin(xyz[:, 2] / (depth + 1e-8))

        # Convert to degrees
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        # Project to image coordinates
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # [0, 1]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # [0, 1]

        # Scale to image size
        proj_x = np.floor(proj_x * self.proj_W).astype(np.int32)
        proj_y = np.floor(proj_y * self.proj_H).astype(np.int32)

        # Clip to valid range
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)

        # Initialize range image
        range_image = np.zeros((self.proj_H, self.proj_W, 5), dtype=np.float32)
        proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=bool)

        # Fill range image (keep closest point per pixel)
        order = np.argsort(depth)[::-1]  # Far to near
        for idx in order:
            px, py = proj_x[idx], proj_y[idx]
            range_image[py, px] = [depth[idx], xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], intensity[idx]]
            proj_mask[py, px] = True

        return range_image, proj_mask

    def _range_image_to_point_cloud(self,
                                     predictions: np.ndarray,
                                     range_image: np.ndarray,
                                     proj_mask: np.ndarray,
                                     original_points: np.ndarray) -> np.ndarray:
        """
        Map range image predictions back to point cloud

        Args:
            predictions: (H, W) predicted labels
            range_image: (H, W, 5) range image
            proj_mask: (H, W) valid pixel mask
            original_points: Original point cloud (N, 3+)

        Returns:
            labels: (N,) predicted labels for each point
        """
        # Extract valid predictions
        valid_preds = predictions[proj_mask]
        valid_xyz = range_image[proj_mask, 1:4]

        # Match back to original points using nearest neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(valid_xyz)
        _, indices = tree.query(original_points[:, :3], k=1)

        labels = valid_preds[indices]
        return labels

    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        """
        Preprocess point cloud to range image

        Args:
            points: Point cloud (N, 4)

        Returns:
            range_tensor: (1, 5, H, W) tensor
        """
        range_image, _ = self._point_cloud_to_range_image(points)

        # Normalize range image
        # Range channel
        range_image[:, :, 0] = np.tanh(range_image[:, :, 0])

        # Convert to tensor: (H, W, C) -> (C, H, W)
        range_tensor = torch.from_numpy(range_image).float()
        range_tensor = range_tensor.permute(2, 0, 1)  # (C, H, W)
        range_tensor = range_tensor.unsqueeze(0)  # (1, C, H, W)

        return range_tensor.to(self.device)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Predict semantic labels using SalsaNext

        Args:
            points: Point cloud (N, 4)

        Returns:
            labels: (N,) predicted labels
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please provide a valid checkpoint.")

        with torch.no_grad():
            # Convert to range image
            range_image, proj_mask = self._point_cloud_to_range_image(points)

            # Preprocess
            range_tensor = self.preprocess(points)

            # Forward pass
            output = self.model(range_tensor)  # (1, num_classes, H, W)

            # Get predictions
            predictions = torch.argmax(output, dim=1)  # (1, H, W)
            predictions = predictions.squeeze(0).cpu().numpy()  # (H, W)

            # Map back to point cloud
            labels = self._range_image_to_point_cloud(
                predictions, range_image, proj_mask, points
            )

        return labels


class RandLANetWrapper(BaseSegmentationModel):
    """
    Wrapper for RandLA-Net model

    To use this:
    1. Clone RandLA-Net repo: https://github.com/QingyongHu/RandLA-Net
    2. Follow their setup instructions
    3. Place pretrained weights in models/weights/
    4. Update the import and initialization below
    """

    def __init__(self, num_classes: int = 20, device: str = 'cuda'):
        super().__init__(num_classes, device)
        print("RandLA-Net wrapper - requires separate installation")
        print("See: https://github.com/QingyongHu/RandLA-Net")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load RandLA-Net model"""
        raise NotImplementedError(
            "RandLA-Net requires separate installation. "
            "Clone the repo and follow setup instructions."
        )

    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess for RandLA-Net"""
        raise NotImplementedError("Implement based on RandLA-Net requirements")

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Predict using RandLA-Net"""
        raise NotImplementedError("Implement based on RandLA-Net inference")


# Factory function for easy model creation
def create_segmentation_model(model_type: str = 'heuristic',
                               num_classes: int = 20,
                               device: str = 'cuda',
                               checkpoint_path: Optional[str] = None) -> BaseSegmentationModel:
    """
    Factory function to create segmentation models

    Args:
        model_type: Type of model ('heuristic', 'simple', 'salsanext', 'randlanet')
        num_classes: Number of semantic classes
        device: Device to run on
        checkpoint_path: Path to model checkpoint

    Returns:
        Segmentation model instance
    """
    if model_type == 'heuristic':
        model = HeuristicSegmentationModel(num_classes=num_classes, device=device)
    elif model_type == 'simple':
        model = SimpleSegmentationModel(num_classes=num_classes, device=device)
    elif model_type == 'salsanext':
        model = SalsaNextWrapper(num_classes=num_classes, device=device)
    elif model_type == 'randlanet':
        model = RandLANetWrapper(num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_model(checkpoint_path)
    return model
