"""
Main Demo Pipeline for SLAM Dataset Annotation

This script demonstrates the complete pipeline:
1. Download KITTI dataset
2. Load and preprocess point clouds
3. Run segmentation model
4. Extract objects
5. Generate NLP annotations
6. Visualize results
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

from download_kitti import KITTIDownloader
from utils.point_cloud_utils import PointCloudProcessor, KITTIDataLoader
from utils.visualization import Visualizer
from utils.interactive_viewer import InteractiveAnnotationViewer
from models.segmentation_model import create_segmentation_model
from models.nlp_model import create_nlp_model


class AnnotationPipeline:
    """Complete pipeline for dataset annotation"""

    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """
        Initialize pipeline with configuration

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup paths
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.processor = PointCloudProcessor()
        self.visualizer = Visualizer()

        # Initialize models
        print("\n" + "="*50)
        print("Initializing Models")
        print("="*50)

        self.seg_model = create_segmentation_model(
            model_type=self.config['segmentation']['model_type'],
            num_classes=self.config['segmentation']['num_classes'],
            device=self.config['processing']['device'],
            checkpoint_path=self.config['segmentation']['checkpoint_path']
        )

        self.nlp_model = create_nlp_model(
            model_type=self.config['nlp']['model_type'],
            device=self.config['processing']['device'],
            checkpoint_path=self.config['nlp']['checkpoint_path']
        )

        print("\nModels initialized successfully!")

    def download_dataset(self):
        """Download KITTI dataset if not already present"""
        print("\n" + "="*50)
        print("Dataset Download")
        print("="*50)

        data_root = Path(self.config['dataset']['root_dir'])

        # Check if dataset already exists
        sequences_dir = data_root / 'sequences'
        if sequences_dir.exists():
            print(f"Dataset already exists at {data_root}")
            return

        # Download dataset
        downloader = KITTIDownloader(root_dir=str(data_root))
        components = self.config['dataset'].get('download_components', None)
        downloader.download_dataset(components=components)
        downloader.organize_dataset()

    def preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply preprocessing to point cloud"""
        cfg = self.config['preprocessing']

        # Filter by range
        points = self.processor.filter_points(
            points,
            x_range=(cfg['filter_range']['x_min'], cfg['filter_range']['x_max']),
            y_range=(cfg['filter_range']['y_min'], cfg['filter_range']['y_max']),
            z_range=(cfg['filter_range']['z_min'], cfg['filter_range']['z_max'])
        )

        # Voxel downsampling
        if cfg.get('voxel_downsample', False):
            points = self.processor.voxel_downsample(
                points,
                voxel_size=cfg['voxel_size']
            )

        return points

    def process_scan(self, points: np.ndarray, scan_id: int):
        """
        Process a single scan through the complete pipeline

        Args:
            points: Point cloud array
            scan_id: Scan identifier

        Returns:
            Dictionary with results
        """
        print(f"\nProcessing scan {scan_id}")
        print(f"  Original points: {len(points)}")

        # Preprocess
        points_processed = self.preprocess_point_cloud(points)
        print(f"  After preprocessing: {len(points_processed)}")

        # Segmentation
        print("  Running segmentation...")
        labels = self.seg_model.predict(points_processed)
        print(f"  Detected {len(np.unique(labels))} unique classes")

        # Show class distribution with colors
        class_dist = self.seg_model.get_class_distribution(labels, colored=True)
        print("  Class distribution:")
        # Sort by count (descending)
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1]['count'], reverse=True)
        for label_id, info in sorted_classes:
            percentage = 100.0 * info['count'] / len(labels)
            # Create a colored bar
            bar_length = int(percentage / 2)  # Scale to fit terminal
            bar = '█' * bar_length
            colored_bar = f"{info['color']}{bar}\033[0m"
            print(f"    {info['colored_name']}: {info['count']:6d} points ({percentage:5.1f}%) {colored_bar}")

        # Extract objects
        print("  Extracting objects...")
        min_points = self.config['segmentation']['min_object_points']
        objects = self.seg_model.extract_objects(
            points_processed,
            labels,
            min_points=min_points
        )
        print(f"  Extracted {len(objects)} objects")

        # Show object breakdown by class
        if len(objects) > 0:
            obj_by_class = {}
            for obj in objects:
                class_name = obj['semantic_class']
                obj_by_class[class_name] = obj_by_class.get(class_name, 0) + 1

            print("  Objects by class:")
            for class_name, count in sorted(obj_by_class.items(), key=lambda x: x[1], reverse=True):
                # Find the label ID for this class
                label_id = next((k for k, v in self.seg_model.class_names.items() if v == class_name), 0)
                colored_name = self.seg_model.get_colored_class_name(label_id)
                print(f"    {colored_name}: {count} objects")

        # Generate annotations
        if self.config['nlp']['generate_object_descriptions']:
            print("  Generating object descriptions...")
            objects = self.nlp_model.annotate_objects(objects)

        # Generate scene description
        scene_description = None
        if self.config['nlp']['generate_scene_description']:
            print("  Generating scene description...")
            scene_description = self.nlp_model.describe_scene(objects)

        return {
            'scan_id': scan_id,
            'points': points_processed,
            'labels': labels,
            'objects': objects,
            'scene_description': scene_description
        }

    def visualize_results(self, results: dict):
        """Visualize processing results"""
        if not self.config['visualization']['enabled']:
            return

        scan_id = results['scan_id']
        points = results['points']
        labels = results['labels']
        objects = results['objects']
        scene_description = results['scene_description']

        # Get class names from the segmentation model
        class_names = self.seg_model.class_names

        viz_output_dir = self.output_dir / 'visualizations'
        viz_output_dir.mkdir(exist_ok=True)

        # 2D visualization
        if self.config['visualization']['show_2d']:
            save_path = viz_output_dir / f'scan_{scan_id:06d}_2d.png' \
                if self.config['visualization']['save_figures'] else None

            self.visualizer.plot_point_cloud_2d(
                points,
                labels,
                title=f"Scan {scan_id:06d} - Bird's Eye View",
                save_path=str(save_path) if save_path else None,
                class_names=class_names
            )

        # 3D matplotlib plot
        if self.config['visualization']['save_figures']:
            save_path_3d = viz_output_dir / f'scan_{scan_id:06d}_3d.png'
            self.visualizer.plot_point_cloud_3d(
                points,
                labels,
                title=f"Scan {scan_id:06d} - 3D View",
                save_path=str(save_path_3d),
                class_names=class_names
            )

        # Interactive 3D visualization with NLP annotations
        if self.config['visualization']['show_3d']:
            print("\n  Launching interactive 3D viewer...")
            print("  - Double-click on objects to see NLP annotations")
            print("  - Descriptions include object class, surroundings, and physical properties")
            print("  - Green lines show spatial relationships to nearby objects")

            viewer = InteractiveAnnotationViewer(
                points=points,
                labels=labels,
                objects=objects,
                scene_description=scene_description,
                nlp_model=self.nlp_model
            )
            viewer.visualize()

    def save_results(self, results: dict):
        """Save processing results to disk"""
        if not self.config['output']['save_annotations']:
            return

        scan_id = results['scan_id']
        annotations_dir = self.output_dir / 'annotations'
        annotations_dir.mkdir(exist_ok=True)

        # Prepare data for serialization
        output_data = {
            'scan_id': scan_id,
            'num_points': len(results['points']),
            'num_objects': len(results['objects']),
            'scene_description': results['scene_description'],
            'objects': []
        }

        # Add object information
        for obj in results['objects']:
            obj_data = {
                'instance_id': obj['instance_id'],
                'semantic_label': obj['semantic_label'],
                'num_points': obj['num_points'],
                'centroid': obj['centroid'].tolist(),
                'bbox_min': obj['bbox_min'].tolist(),
                'bbox_max': obj['bbox_max'].tolist(),
                'description': obj.get('description', '')
            }
            output_data['objects'].append(obj_data)

        # Save as JSON
        output_file = annotations_dir / f'scan_{scan_id:06d}.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  Saved annotations to {output_file}")

        # Save segmented objects if requested
        if self.config['output']['save_segmented_objects']:
            objects_dir = self.output_dir / 'objects' / f'scan_{scan_id:06d}'
            objects_dir.mkdir(parents=True, exist_ok=True)

            for obj in results['objects']:
                obj_file = objects_dir / f'object_{obj["instance_id"]:03d}.npy'
                np.save(obj_file, obj['points'])

    def run(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print(" SLAM Dataset Annotation Pipeline")
        print("="*70)

        # Step 1: Download dataset
        self.download_dataset()

        # Step 2: Load dataset
        print("\n" + "="*50)
        print("Loading Dataset")
        print("="*50)

        data_loader = KITTIDataLoader(
            data_root=self.config['dataset']['root_dir'],
            sequence=self.config['dataset']['sequence']
        )

        print(f"Loaded sequence {self.config['dataset']['sequence']}")
        print(f"Total scans: {len(data_loader)}")

        # Step 3: Process scans
        print("\n" + "="*50)
        print("Processing Point Clouds")
        print("="*50)

        num_scans = self.config['processing']['num_scans']
        if num_scans < 0:
            num_scans = len(data_loader)

        start_idx = self.config['processing']['start_index']
        end_idx = min(start_idx + num_scans, len(data_loader))

        for idx in tqdm(range(start_idx, end_idx), desc="Processing scans"):
            # Load point cloud
            points = data_loader[idx]

            # Process through pipeline
            results = self.process_scan(points, scan_id=idx)

            # Print scene description
            if results['scene_description']:
                print(f"\n  Scene Description:")
                print(f"  {'-'*60}")
                for line in results['scene_description'].split('\n'):
                    print(f"  {line}")
                print(f"  {'-'*60}")

            # Print some object descriptions
            if results['objects'] and self.config['nlp']['generate_object_descriptions']:
                print(f"\n  Sample Object Descriptions:")
                for i, obj in enumerate(results['objects'][:3]):  # Show first 3
                    print(f"  [{i+1}] {obj.get('description', 'N/A')}")

            # Visualize
            self.visualize_results(results)

            # Save results
            self.save_results(results)

        print("\n" + "="*50)
        print("Pipeline Complete!")
        print("="*50)
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='SLAM Dataset Annotation Pipeline Demo'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download dataset and exit'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AnnotationPipeline(config_path=args.config)

    if args.download_only:
        pipeline.download_dataset()
        return

    if args.skip_download:
        # Modify config to skip download check
        pass

    # Run complete pipeline
    pipeline.run()


if __name__ == '__main__':
    main()
