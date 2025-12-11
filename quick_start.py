"""
Quick Start Script - Minimal Setup Demo

This script provides a minimal working example that:
1. Downloads a small portion of KITTI dataset
2. Processes a few scans
3. Demonstrates the complete pipeline

Use this to quickly test the setup before running the full pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.point_cloud_utils import PointCloudProcessor
from utils.visualization import Visualizer
from models.segmentation_model import create_segmentation_model
from models.nlp_model import create_nlp_model
import yaml


def create_synthetic_data():
    """Create synthetic point cloud data for testing"""
    print("Creating synthetic point cloud data for testing...")

    # Create a simple scene with ground, a box, and scattered points
    points_list = []

    # Ground plane
    x = np.random.uniform(-20, 20, 1000)
    y = np.random.uniform(-20, 20, 1000)
    z = np.random.normal(-1.5, 0.1, 1000)
    intensity = np.random.uniform(0.3, 0.5, 1000)
    ground = np.column_stack([x, y, z, intensity])
    points_list.append(ground)

    # Box (representing a car)
    box_x = np.random.uniform(5, 8, 500)
    box_y = np.random.uniform(-2, 2, 500)
    box_z = np.random.uniform(-1, 1, 500)
    intensity = np.random.uniform(0.6, 0.8, 500)
    box = np.column_stack([box_x, box_y, box_z, intensity])
    points_list.append(box)

    # Another box (another vehicle)
    box2_x = np.random.uniform(-8, -5, 400)
    box2_y = np.random.uniform(3, 6, 400)
    box2_z = np.random.uniform(-1, 1, 400)
    intensity = np.random.uniform(0.5, 0.7, 400)
    box2 = np.column_stack([box2_x, box2_y, box2_z, intensity])
    points_list.append(box2)

    # Vertical structure (pole or tree)
    pole_x = np.random.normal(10, 0.2, 200)
    pole_y = np.random.normal(-10, 0.2, 200)
    pole_z = np.random.uniform(-1, 4, 200)
    intensity = np.random.uniform(0.4, 0.6, 200)
    pole = np.column_stack([pole_x, pole_y, pole_z, intensity])
    points_list.append(pole)

    # Combine all points
    points = np.vstack(points_list)

    # Create pseudo-labels for demonstration
    labels = np.zeros(len(points), dtype=np.int32)
    labels[:1000] = 9  # ground = road
    labels[1000:1500] = 1  # first box = car
    labels[1500:1900] = 4  # second box = truck
    labels[1900:] = 18  # pole

    return points, labels


def quick_demo():
    """Run a quick demonstration of the pipeline"""
    print("="*70)
    print(" Quick Start Demo - SLAM Dataset Annotation Pipeline")
    print("="*70)

    # Create output directory
    output_dir = Path('./outputs/quick_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create synthetic data
    print("\n[1/5] Creating synthetic point cloud data...")
    points, true_labels = create_synthetic_data()
    print(f"  Generated {len(points)} points")

    # Step 2: Load configuration and initialize models
    print("\n[2/5] Loading configuration and initializing models...")

    # Load config
    config_path = Path('./configs/default_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("  Warning: Config file not found, using defaults")
        config = {
            'segmentation': {'model_type': 'salsanext', 'num_classes': 20, 'checkpoint_path': 'models/weights/SalsaNext.pth', 'min_object_points': 150},
            'nlp': {'model_type': 'llama', 'checkpoint_path': None},
            'processing': {'device': 'cuda'}
        }

    # Initialize components
    processor = PointCloudProcessor()
    visualizer = Visualizer()

    # Create segmentation model
    seg_config = config.get('segmentation', {})
    device = config.get('processing', {}).get('device', 'cpu')

    seg_model = create_segmentation_model(
        model_type=seg_config.get('model_type', 'salsanext'),
        num_classes=seg_config.get('num_classes', 20),
        device=device,
        checkpoint_path=seg_config.get('checkpoint_path')
    )

    # Create NLP model
    nlp_config = config.get('nlp', {})
    nlp_model = create_nlp_model(
        model_type=nlp_config.get('model_type', 'llama'),
        device=device,
        checkpoint_path=nlp_config.get('checkpoint_path')
    )

    print(f"  Segmentation model: {seg_config.get('model_type', 'salsanext')}")
    print(f"  NLP model: {nlp_config.get('model_type', 'llama')}")
    print(f"  Device: {device}")

    # Step 3: Preprocess
    print("\n[3/5] Preprocessing point cloud...")
    points_filtered = processor.filter_points(
        points,
        x_range=(-30, 30),
        y_range=(-30, 30),
        z_range=(-3, 10)
    )
    print(f"  After filtering: {len(points_filtered)} points")

    # Step 4: Segmentation
    print("\n[4/5] Running segmentation...")
    print("  Note: Using untrained model - predictions will be random")
    predicted_labels = seg_model.predict(points_filtered)

    # For demo purposes, use the true labels to show what good results look like
    print("  Using ground truth labels for demonstration...")
    labels_to_use = true_labels[:len(points_filtered)]

    # Extract objects
    print("  Extracting objects...")
    min_points = config.get('segmentation', {}).get('min_object_points', 150)
    clustering_config = config.get('segmentation', {}).get('clustering', None)
    merging_config = config.get('segmentation', {}).get('merging', None)
    relabel_merged = config.get('segmentation', {}).get('relabel_merged', True)
    objects = seg_model.extract_objects(
        points_filtered,
        labels_to_use,
        min_points=min_points,
        clustering_config=clustering_config,
        merging_config=merging_config,
        relabel_merged=relabel_merged
    )
    print(f"  Extracted {len(objects)} objects")

    # Step 5: NLP Annotation
    print("\n[5/5] Generating annotations...")
    objects = nlp_model.annotate_objects(objects)
    scene_description = nlp_model.describe_scene(objects)

    # Display results
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)

    print("\nScene Description:")
    print("-" * 70)
    print(scene_description)
    print("-" * 70)

    print("\nDetected Objects:")
    print("-" * 70)
    for i, obj in enumerate(objects):
        print(f"\n[Object {i+1}]")
        print(f"  {obj['description']}")
        print(f"  Points: {obj['num_points']}")
        print(f"  Position: ({obj['centroid'][0]:.1f}, {obj['centroid'][1]:.1f}, {obj['centroid'][2]:.1f})")

    # Visualization
    print("\n" + "="*70)
    print(" VISUALIZATION")
    print("="*70)

    print("\nGenerating visualizations...")

    # Get class names from segmentation model for legend
    class_names = seg_model.class_names

    # 2D Bird's eye view
    save_path_2d = output_dir / 'scene_birdseye.png'
    visualizer.visualize_segmentation_results(
        points_filtered,
        labels_to_use,
        save_path=str(save_path_2d)
    )

    # 3D visualization
    save_path_3d = output_dir / 'scene_3d.png'
    print("\nGenerating 3D visualization...")
    visualizer.plot_point_cloud_3d(
        points_filtered,
        labels_to_use,
        title="Quick Demo - Segmented Point Cloud (3D View)",
        save_path=str(save_path_3d),
        class_names=class_names
    )

    # Interactive 3D viewer (Open3D)
    print("\nLaunching interactive 3D viewer with improved controls...")
    visualizer.visualize_with_open3d(
        points_filtered,
        labels_to_use,
        window_name="Quick Demo - Interactive 3D Viewer",
        class_names=class_names,
        show_legend=True
    )

    print(f"\nVisualizations saved to: {output_dir}")

    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"  Processed points: {len(points_filtered)}")
    print(f"  Detected objects: {len(objects)}")
    print(f"  Output directory: {output_dir}")
    print("\nQuick demo complete!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run full demo: python demo.py")
    print("  3. Download real KITTI data: python download_kitti.py")
    print("="*70)


if __name__ == '__main__':
    try:
        quick_demo()
    except Exception as e:
        print(f"\nError during quick demo: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
