"""
Visualization utilities for point clouds and segmentation results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import open3d as o3d
from typing import Optional, List, Tuple, Dict


class Visualizer:
    """Visualization tools for point clouds and segmentation results"""

    # Color map for semantic classes (can be customized)
    SEMANTIC_COLORS = {
        0: [0, 0, 0],         # unlabeled - black
        1: [245, 150, 100],   # car - coral
        2: [245, 230, 100],   # bicycle - yellow
        3: [150, 60, 30],     # motorcycle - brown
        4: [180, 30, 80],     # truck - purple
        5: [255, 0, 0],       # other-vehicle - red
        6: [30, 30, 255],     # person - blue
        7: [200, 40, 255],    # bicyclist - pink
        8: [90, 30, 150],     # motorcyclist - dark purple
        9: [255, 0, 255],     # road - magenta
        10: [255, 150, 255],  # parking - light magenta
        11: [75, 0, 75],      # sidewalk - dark magenta
        12: [75, 0, 175],     # other-ground - indigo
        13: [0, 200, 255],    # building - cyan
        14: [50, 120, 255],   # fence - light blue
        15: [0, 175, 0],      # vegetation - green
        16: [0, 60, 135],     # trunk - dark blue
        17: [80, 240, 150],   # terrain - light green
        18: [150, 240, 255],  # pole - very light blue
        19: [0, 0, 255],      # traffic-sign - bright blue
    }

    @staticmethod
    def plot_point_cloud_2d(points: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            title: str = "Point Cloud (Bird's Eye View)",
                            save_path: Optional[str] = None,
                            class_names: Optional[Dict[int, str]] = None):
        """
        Plot point cloud in 2D bird's eye view

        Args:
            points: Point cloud array (N, 3+)
            colors: Color for each point (N, 3) or (N,) for labels
            title: Plot title
            save_path: Path to save figure
            class_names: Dictionary mapping label IDs to class names
        """
        fig, ax = plt.subplots(figsize=(14, 12))

        if colors is not None and len(colors.shape) == 1:
            # Colors are labels, map to RGB
            unique_labels = np.unique(colors)

            # Map labels to colors using SEMANTIC_COLORS
            label_colors = np.zeros((len(colors), 3))
            for label in unique_labels:
                mask = colors == label
                if label in Visualizer.SEMANTIC_COLORS:
                    label_colors[mask] = np.array(Visualizer.SEMANTIC_COLORS[label]) / 255.0
                else:
                    np.random.seed(int(label))
                    label_colors[mask] = np.random.rand(3)

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=label_colors,
                s=1,
                alpha=0.6
            )

            # Create custom legend with class names and colors
            if class_names:
                legend_elements = []
                for label in sorted(unique_labels):
                    if label in Visualizer.SEMANTIC_COLORS:
                        color = np.array(Visualizer.SEMANTIC_COLORS[label]) / 255.0
                    else:
                        np.random.seed(int(label))
                        color = np.random.rand(3)

                    class_name = class_names.get(int(label), f'class_{int(label)}')
                    legend_elements.append(
                        Patch(facecolor=color, label=f'{class_name} ({int(label)})')
                    )

                ax.legend(handles=legend_elements, loc='upper left',
                         bbox_to_anchor=(1.02, 1), fontsize=9,
                         title='Semantic Classes')
        elif colors is not None:
            # Colors are RGB values
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=colors / 255.0 if colors.max() > 1 else colors,
                s=1,
                alpha=0.6
            )
        else:
            # No colors, use height as color
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=points[:, 2],
                s=1,
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label='Height (m)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_point_cloud_3d(points: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            title: str = "Point Cloud 3D View",
                            save_path: Optional[str] = None,
                            class_names: Optional[Dict[int, str]] = None):
        """
        Plot point cloud in 3D

        Args:
            points: Point cloud array (N, 3+)
            colors: Color for each point (N, 3) or (N,) for labels
            title: Plot title
            save_path: Path to save figure
            class_names: Dictionary mapping label IDs to class names
        """
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Downsample for visualization if too many points
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points_vis = points[indices]
            colors_vis = colors[indices] if colors is not None else None
        else:
            points_vis = points
            colors_vis = colors

        if colors_vis is not None and len(colors_vis.shape) == 1:
            # Colors are labels, map to RGB
            unique_labels = np.unique(colors_vis)

            # Map labels to colors using SEMANTIC_COLORS
            label_colors = np.zeros((len(colors_vis), 3))
            for label in unique_labels:
                mask = colors_vis == label
                if label in Visualizer.SEMANTIC_COLORS:
                    label_colors[mask] = np.array(Visualizer.SEMANTIC_COLORS[label]) / 255.0
                else:
                    np.random.seed(int(label))
                    label_colors[mask] = np.random.rand(3)

            ax.scatter(
                points_vis[:, 0],
                points_vis[:, 1],
                points_vis[:, 2],
                c=label_colors,
                s=1,
                alpha=0.6
            )

            # Create custom legend with class names and colors
            if class_names:
                legend_elements = []
                for label in sorted(unique_labels):
                    if label in Visualizer.SEMANTIC_COLORS:
                        color = np.array(Visualizer.SEMANTIC_COLORS[label]) / 255.0
                    else:
                        np.random.seed(int(label))
                        color = np.random.rand(3)

                    class_name = class_names.get(int(label), f'class_{int(label)}')
                    legend_elements.append(
                        Patch(facecolor=color, label=f'{class_name} ({int(label)})')
                    )

                ax.legend(handles=legend_elements, loc='upper left',
                         bbox_to_anchor=(1.02, 1), fontsize=8,
                         title='Semantic Classes')
        elif colors is not None:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors / 255.0 if colors.max() > 1 else colors,
                s=1,
                alpha=0.6
            )
        else:
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=points[:, 2],
                s=1,
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label='Height (m)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_with_open3d(points: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              window_name: str = "Point Cloud Viewer",
                              class_names: Optional[Dict[int, str]] = None,
                              show_legend: bool = True):
        """
        Interactive visualization using Open3D with improved controls and legend

        Args:
            points: Point cloud array (N, 3+)
            labels: Semantic labels (N,)
            window_name: Window title
            class_names: Dictionary mapping label IDs to class names
            show_legend: Whether to print legend and controls to console

        Controls:
            - Mouse Left: Rotate
            - Mouse Right: Pan
            - Mouse Wheel: Zoom
            - Ctrl + Mouse Left: Pan (alternative)
            - Q/ESC: Quit
            - R: Reset view
            - +/-: Increase/decrease point size
            - H: Print help
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Build legend information
        legend_info = []

        if labels is not None:
            # Map labels to colors
            colors = np.zeros((len(labels), 3))
            unique_labels = np.unique(labels)

            for label in unique_labels:
                mask = labels == label
                if label in Visualizer.SEMANTIC_COLORS:
                    color_rgb = np.array(Visualizer.SEMANTIC_COLORS[label]) / 255.0
                    colors[mask] = color_rgb
                else:
                    # Random color for unknown labels
                    np.random.seed(int(label))
                    color_rgb = np.random.rand(3)
                    colors[mask] = color_rgb

                # Build legend entry
                class_name = class_names.get(int(label), f'Class {int(label)}') if class_names else f'Class {int(label)}'
                count = np.sum(mask)
                legend_info.append((label, class_name, color_rgb, count))

            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif points.shape[1] > 3:
            # Use intensity if available
            intensity = points[:, 3]
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            colors = plt.cm.viridis(intensity_norm)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Print legend and controls
        if show_legend:
            print("\n" + "="*70)
            print(" Open3D Point Cloud Viewer - Controls")
            print("="*70)
            print("\n📋 CAMERA CONTROLS:")
            print("  Mouse Left Button + Drag       : Rotate view")
            print("  Mouse Right Button + Drag      : Pan (translate)")
            print("  Mouse Wheel / Middle Button    : Zoom in/out")
            print("  Ctrl + Mouse Left + Drag       : Pan (alternative)")
            print("  Shift + Mouse Left + Drag      : Roll view")
            print("\n⌨️  KEYBOARD SHORTCUTS:")
            print("  Q or ESC                       : Quit viewer")
            print("  R                              : Reset camera to default view")
            print("  +  (or =)                      : Increase point size")
            print("  -  (or _)                      : Decrease point size")
            print("  H                              : Print this help again")
            print("  Double Click on Point          : Set rotation center")

            if legend_info:
                print("\n🎨 COLOR LEGEND:")
                # Sort by label ID
                legend_info.sort(key=lambda x: x[0])
                for label_id, class_name, color, count in legend_info:
                    # Create color block using RGB
                    r, g, b = [int(c * 255) for c in color]
                    color_block = f"\033[48;2;{r};{g};{b}m   \033[0m"
                    percentage = 100.0 * count / len(labels)
                    print(f"  {color_block}  {class_name:20s} ({label_id:2d}): {count:6d} points ({percentage:5.1f}%)")

            print("="*70 + "\n")

        # Create visualizer with custom settings
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1920, height=1080)
        vis.add_geometry(pcd)

        # Set viewing options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.05, 0.05, 0.05])  # Dark gray background
        opt.show_coordinate_frame = True  # Show XYZ axes

        # Set better initial camera view (bird's eye view at 45 degrees)
        ctr = vis.get_view_control()

        # Calculate scene center and extent
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = np.max(extent)

        # Set camera to look at the scene from a good angle
        # Position camera at 45-degree angle, slightly elevated
        distance = max_extent * 2.0
        ctr.set_front([0.5, 0.5, -0.7])  # Look direction
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])  # Z is up
        ctr.set_zoom(0.5)

        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_segmentation_results(points: np.ndarray,
                                       labels: np.ndarray,
                                       class_names: Optional[List[str]] = None,
                                       save_path: Optional[str] = None):
        """
        Create comprehensive visualization of segmentation results

        Args:
            points: Point cloud array (N, 3+)
            labels: Predicted labels (N,)
            class_names: List of class names
            save_path: Path to save visualization
        """
        fig = plt.figure(figsize=(16, 8))

        # Bird's eye view
        ax1 = fig.add_subplot(121)
        unique_labels = np.unique(labels)

        scatter = ax1.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            s=1,
            cmap='tab20',
            alpha=0.6
        )
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title("Segmentation - Bird's Eye View")
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Class distribution
        ax2 = fig.add_subplot(122)
        unique, counts = np.unique(labels, return_counts=True)

        if class_names and len(class_names) >= len(unique):
            labels_for_plot = [class_names[int(u)] if int(u) < len(class_names) else f'Class {int(u)}'
                               for u in unique]
        else:
            labels_for_plot = [f'Class {int(u)}' for u in unique]

        ax2.barh(range(len(unique)), counts)
        ax2.set_yticks(range(len(unique)))
        ax2.set_yticklabels(labels_for_plot)
        ax2.set_xlabel('Number of Points')
        ax2.set_title('Class Distribution')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved segmentation visualization to {save_path}")

        plt.show()

    @staticmethod
    def create_comparison_plot(points: np.ndarray,
                              pred_labels: np.ndarray,
                              gt_labels: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None):
        """
        Create side-by-side comparison of predictions and ground truth

        Args:
            points: Point cloud array (N, 3+)
            pred_labels: Predicted labels (N,)
            gt_labels: Ground truth labels (N,)
            save_path: Path to save figure
        """
        if gt_labels is None:
            print("No ground truth labels provided, showing only predictions")
            Visualizer.plot_point_cloud_2d(points, pred_labels, "Predictions")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Predictions
        ax1.scatter(
            points[:, 0],
            points[:, 1],
            c=pred_labels,
            s=1,
            cmap='tab20',
            alpha=0.6
        )
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Predictions')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Ground Truth
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            c=gt_labels,
            s=1,
            cmap='tab20',
            alpha=0.6
        )
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Ground Truth')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")

        plt.show()
