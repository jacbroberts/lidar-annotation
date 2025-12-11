"""
Interactive 3D Viewer for Annotated Point Clouds

Features:
- Click to select objects
- View NLP annotations for each object
- Visualize relationships between objects
- Color-coded semantic segmentation
"""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from typing import List, Dict, Optional, Tuple


class InteractiveAnnotationViewer:
    """Interactive 3D viewer for point clouds with NLP annotations"""

    SEMANTIC_COLORS = {
        0: [0, 0, 0],         # unlabeled
        1: [245, 150, 100],   # car
        2: [245, 230, 100],   # bicycle
        3: [150, 60, 30],     # motorcycle
        4: [180, 30, 80],     # truck
        5: [255, 0, 0],       # other-vehicle
        6: [30, 30, 255],     # person
        7: [200, 40, 255],    # bicyclist
        8: [90, 30, 150],     # motorcyclist
        9: [255, 0, 255],     # road
        10: [255, 150, 255],  # parking
        11: [75, 0, 75],      # sidewalk
        12: [75, 0, 175],     # other-ground
        13: [0, 200, 255],    # building
        14: [50, 120, 255],   # fence
        15: [0, 175, 0],      # vegetation
        16: [0, 60, 135],     # trunk
        17: [80, 240, 150],   # terrain
        18: [150, 240, 255],  # pole
        19: [0, 0, 255],      # traffic-sign
    }

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 objects: List[Dict], scene_description: Optional[str] = None,
                 nlp_model=None):
        """
        Initialize interactive viewer

        Args:
            points: Point cloud (N, 3+)
            labels: Semantic labels (N,)
            objects: List of extracted objects with annotations
            scene_description: Overall scene description
            nlp_model: NLP model for generating contextual descriptions
        """
        self.points = points
        self.labels = labels
        self.objects = objects
        self.scene_description = scene_description
        self.nlp_model = nlp_model

        # Set all_objects on NLP model for spatial relationship computation
        if self.nlp_model is not None:
            self.nlp_model.all_objects = objects

        self.selected_object_idx = None
        self.geometries = {}  # Store geometries for manipulation

        # Double-click detection
        import time
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 0.3  # seconds

        # Build spatial index for object selection
        self._build_object_index()

    def _build_object_index(self):
        """Build index for fast object lookup"""
        self.object_centroids = np.array([obj['centroid'] for obj in self.objects])
        self.object_bboxes = [(obj['bbox_min'], obj['bbox_max']) for obj in self.objects]

    def _create_point_cloud_geometry(self) -> o3d.geometry.PointCloud:
        """Create colored point cloud geometry with enhanced visibility"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])

        # Map labels to colors with enhanced saturation
        colors = np.zeros((len(self.labels), 3))
        for label in np.unique(self.labels):
            mask = self.labels == label
            if label in self.SEMANTIC_COLORS:
                # Convert to float and enhance saturation
                base_color = np.array(self.SEMANTIC_COLORS[label]) / 255.0
                # Boost color intensity for better visibility
                base_color = np.clip(base_color * 1.2, 0, 1)
                colors[mask] = base_color
            else:
                np.random.seed(int(label))
                colors[mask] = np.random.rand(3)

        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _create_bounding_boxes(self) -> List[o3d.geometry.LineSet]:
        """Create bounding boxes for all objects"""
        boxes = []
        for obj in self.objects:
            box = self._create_bbox_lineset(
                obj['bbox_min'],
                obj['bbox_max'],
                color=[0.7, 0.7, 0.7]  # Gray by default
            )
            boxes.append(box)
        return boxes

    def _create_bbox_lineset(self, bbox_min: np.ndarray, bbox_max: np.ndarray,
                            color: List[float]) -> o3d.geometry.LineSet:
        """Create a bounding box as a line set"""
        # Define 8 corners of the bbox
        corners = np.array([
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
        ])

        # Define 12 edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        return line_set

    def _create_text_label_3d(self, position: np.ndarray, text: str) -> o3d.geometry.TriangleMesh:
        """
        Create a text label in 3D space (using a sphere as placeholder)
        Note: Open3D doesn't natively support 3D text, so we use markers
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere.compute_vertex_normals()  # Compute normals to fix lighting warnings
        sphere.translate(position)
        sphere.paint_uniform_color([1, 0.706, 0])  # Orange marker
        return sphere

    def _find_object_at_point(self, point: np.ndarray) -> Optional[int]:
        """Find which object contains or is closest to the given point"""
        # Find closest object by centroid distance
        if len(self.object_centroids) == 0:
            return None

        distances = np.linalg.norm(self.object_centroids - point, axis=1)
        closest_idx = np.argmin(distances)

        # Check if point is within bbox
        bbox_min, bbox_max = self.object_bboxes[closest_idx]
        if (np.all(point >= bbox_min) and np.all(point <= bbox_max)) or distances[closest_idx] < 2.0:
            return closest_idx

        return None

    def _compute_object_relationships(self, obj_idx: int) -> List[Tuple[int, str, str]]:
        """
        Compute spatial relationships between selected object and others

        Returns:
            List of (object_index, relationship_description, class_name) tuples
        """
        relationships = []
        selected_centroid = self.objects[obj_idx]['centroid']
        selected_class = self.objects[obj_idx]['semantic_class']

        for i, other_obj in enumerate(self.objects):
            if i == obj_idx:
                continue

            other_centroid = other_obj['centroid']
            distance = np.linalg.norm(selected_centroid - other_centroid)

            # Only show nearby objects
            if distance < 15.0:  # Within 15 meters
                # Determine spatial relationship
                direction = other_centroid - selected_centroid

                # Horizontal relationship
                if abs(direction[0]) > abs(direction[1]):
                    horiz = "right of" if direction[0] > 0 else "left of"
                else:
                    horiz = "in front of" if direction[1] > 0 else "behind"

                relationship = f"{distance:.1f}m {horiz}"
                relationships.append((i, relationship, other_obj['semantic_class']))

        # Sort by distance
        relationships.sort(key=lambda x: float(x[1].split('m')[0]))
        return relationships

    def _generate_contextual_description(self, obj_idx: int) -> str:
        """
        Generate a contextual description for the selected object
        including information about its surroundings
        """
        obj = self.objects[obj_idx]
        obj_class = obj['semantic_class']

        # Get nearby objects
        relationships = self._compute_object_relationships(obj_idx)

        # Build context description
        description_parts = []

        # Use NLP model to generate intelligent description
        if self.nlp_model is not None:
            print(f"Generating NLP description for {obj_class}...")
            nlp_desc = self.nlp_model.describe_object(obj)
            description_parts.append(f"NLP Description:\n{nlp_desc}")
        else:
            # Fallback if no NLP model
            base_desc = obj.get('description', f"A {obj_class} detected in the scene")
            description_parts.append(base_desc)

        # Add spatial context
        if relationships:
            nearby_summary = []
            for i, (rel_idx, rel_desc, rel_class) in enumerate(relationships[:3]):  # Top 3
                nearby_summary.append(f"{rel_class} {rel_desc}")

            if nearby_summary:
                description_parts.append(f"\n\nNearby objects:")
                for item in nearby_summary:
                    description_parts.append(f"  - {item}")

        # Add physical properties
        bbox_size = obj['bbox_max'] - obj['bbox_min']
        description_parts.append(f"\n\nPhysical properties:")
        description_parts.append(f"  - Size: {bbox_size[0]:.1f}m x {bbox_size[1]:.1f}m x {bbox_size[2]:.1f}m")
        description_parts.append(f"  - Points: {obj['num_points']:,}")

        return "\n".join(description_parts)

    def visualize(self):
        """Launch interactive visualization"""
        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("NLP-Annotated Point Cloud Viewer", 1600, 1000)

        # Create scene widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        # Set dark background (matching quick_start.py viewer)
        self.scene.scene.set_background([0.05, 0.05, 0.05, 1.0])  # Dark gray background

        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        coord_mat = rendering.MaterialRecord()
        coord_mat.shader = "defaultUnlit"
        self.scene.scene.add_geometry("coordinate_frame", coord_frame, coord_mat)

        # Add point cloud with larger points
        pcd = self._create_point_cloud_geometry()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0  # Larger points for better visibility
        self.scene.scene.add_geometry("point_cloud", pcd, mat)

        # Add bounding boxes with thicker lines
        self.bbox_linesets = self._create_bounding_boxes()
        bbox_mat = rendering.MaterialRecord()
        bbox_mat.shader = "defaultUnlit"
        bbox_mat.line_width = 2.0  # Thicker lines for better visibility
        for i, bbox in enumerate(self.bbox_linesets):
            self.scene.scene.add_geometry(f"bbox_{i}", bbox, bbox_mat)

        # Add labels (markers) for objects
        for i, obj in enumerate(self.objects):
            if obj.get('description'):
                marker = self._create_text_label_3d(obj['centroid'], obj['description'])
                mat_marker = rendering.MaterialRecord()
                mat_marker.shader = "defaultLit"
                self.scene.scene.add_geometry(f"marker_{i}", marker, mat_marker)

        # Setup camera with better initial view
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        # Create info panel
        self.info_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Scene description
        if self.scene_description:
            scene_label = gui.Label("Scene Description:")
            self.info_panel.add_child(scene_label)

            scene_text = gui.Label(self.scene_description[:200] + "...")
            self.info_panel.add_child(scene_text)
            self.info_panel.add_fixed(10)

        # Instructions
        instructions = gui.Label(
            "INSTRUCTIONS:\n"
            "- Rotate: Left mouse drag\n"
            "- Pan: Ctrl + Left drag\n"
            "- Zoom: Scroll wheel\n"
            "- Select: Double-click OR Press N/P\n"
            "- N: Next object\n"
            "- P: Previous object"
        )
        self.info_panel.add_child(instructions)
        self.info_panel.add_fixed(10)

        # Add color legend (simplified - no unicode)
        legend_header = gui.Label("CLASS DISTRIBUTION:")
        self.info_panel.add_child(legend_header)
        self.info_panel.add_fixed(3)

        # Build color legend from unique labels
        unique_labels = np.unique(self.labels)
        label_counts = {}
        for label in unique_labels:
            count = np.sum(self.labels == label)
            label_counts[label] = count

        # Create legend text with class names and counts
        from models.segmentation_model import KITTI_CLASS_NAMES
        legend_text = ""
        # Sort by percentage (descending) for better readability
        sorted_labels = sorted(unique_labels, key=lambda l: label_counts[l], reverse=True)
        for label in sorted_labels:
            class_name = KITTI_CLASS_NAMES.get(int(label), f'class_{int(label)}')
            count = label_counts[label]
            percentage = 100.0 * count / len(self.labels)
            # Simple text format without unicode
            legend_text += f"{class_name}: {percentage:.1f}%\n"

        legend_label = gui.Label(legend_text)
        self.info_panel.add_child(legend_label)
        self.info_panel.add_fixed(10)

        # Separator
        separator_legend = gui.Label("------------------------------")
        self.info_panel.add_child(separator_legend)
        self.info_panel.add_fixed(5)

        # Selected object header
        self.selected_header = gui.Label("=== SELECTED OBJECT ===")
        self.info_panel.add_child(self.selected_header)
        self.info_panel.add_fixed(5)

        # Object basic info
        self.object_info_label = gui.Label("Double-click an object to see details")
        self.info_panel.add_child(self.object_info_label)
        self.info_panel.add_fixed(10)

        # Separator
        separator1 = gui.Label("----------------------")
        self.info_panel.add_child(separator1)
        self.info_panel.add_fixed(5)

        # NLP Description section
        desc_header = gui.Label("NLP DESCRIPTION:")
        self.info_panel.add_child(desc_header)
        self.info_panel.add_fixed(3)

        # Object description (scrollable if needed)
        self.description_label = gui.Label("")
        self.info_panel.add_child(self.description_label)
        self.info_panel.add_fixed(10)

        # Separator
        separator2 = gui.Label("----------------------")
        self.info_panel.add_child(separator2)
        self.info_panel.add_fixed(5)

        # Spatial relationships
        rel_header = gui.Label("SPATIAL RELATIONSHIPS:")
        self.info_panel.add_child(rel_header)
        self.info_panel.add_fixed(3)

        self.relationships_label = gui.Label("")
        self.info_panel.add_child(self.relationships_label)

        # Object count summary
        summary_text = f"\nTotal Objects: {len(self.objects)}\n"
        obj_by_class = {}
        for obj in self.objects:
            cls = obj['semantic_class']
            obj_by_class[cls] = obj_by_class.get(cls, 0) + 1

        for cls, count in sorted(obj_by_class.items(), key=lambda x: x[1], reverse=True):
            summary_text += f"  - {cls}: {count}\n"

        summary_label = gui.Label(summary_text)
        self.info_panel.add_child(summary_label)

        # Layout
        self.window.add_child(self.scene)
        self.window.add_child(self.info_panel)

        # Set layout
        self.window.set_on_layout(self._on_layout)
        self.scene.set_on_mouse(self._on_mouse)
        self.window.set_on_key(self._on_key)

        print("\n" + "="*60)
        print("KEYBOARD SHORTCUTS:")
        print("  N - Select Next object")
        print("  P - Select Previous object")
        print("  1-9 - Select object by number")
        print("="*60 + "\n")

        # Run
        app.run()

    def _on_layout(self, layout_context):
        """Handle window layout"""
        r = self.window.content_rect
        panel_width = 350

        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.info_panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    def _on_mouse(self, event):
        """Handle mouse events for object selection"""
        import time

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            current_time = time.time()
            current_pos = (event.x, event.y)

            print(f"Mouse click detected at {current_pos}")

            # Check for double-click
            is_double_click = False
            if self.last_click_pos is not None:
                time_diff = current_time - self.last_click_time
                pos_diff = ((current_pos[0] - self.last_click_pos[0])**2 +
                           (current_pos[1] - self.last_click_pos[1])**2)**0.5

                if time_diff < self.double_click_threshold and pos_diff < 10:
                    is_double_click = True
                    print("Double-click detected!")

            self.last_click_time = current_time
            self.last_click_pos = current_pos

            if is_double_click:
                # Pick object on double-click
                x = event.x - self.scene.frame.x
                y = event.y - self.scene.frame.y

                print(f"Picking at scene coordinates ({x}, {y})")

                # Simple approach: find closest object centroid to click
                # Project all object centroids to screen space and find closest
                best_obj_idx = None
                best_distance = float('inf')

                for i, obj in enumerate(self.objects):
                    centroid_3d = obj['centroid']
                    # Simple heuristic - just find closest object by centroid
                    # In a real implementation, we'd project to screen space
                    dist = np.linalg.norm(centroid_3d[:2])  # Distance in XY plane
                    if dist < best_distance:
                        best_distance = dist
                        best_obj_idx = i

                # For now, just cycle through objects on double-click
                if best_obj_idx is None and len(self.objects) > 0:
                    if self.selected_object_idx is None:
                        best_obj_idx = 0
                    else:
                        best_obj_idx = (self.selected_object_idx + 1) % len(self.objects)

                print(f"Selected object index: {best_obj_idx}")

                if best_obj_idx is not None:
                    self._select_object(best_obj_idx)

        return True

    def _on_key(self, event):
        """Handle keyboard events for object selection"""
        if event.type == gui.KeyEvent.Type.DOWN:
            # N - Next object
            if event.key == gui.KeyName.N:
                if len(self.objects) > 0:
                    if self.selected_object_idx is None:
                        next_idx = 0
                    else:
                        next_idx = (self.selected_object_idx + 1) % len(self.objects)
                    self._select_object(next_idx)

            # P - Previous object
            elif event.key == gui.KeyName.P:
                if len(self.objects) > 0:
                    if self.selected_object_idx is None:
                        prev_idx = len(self.objects) - 1
                    else:
                        prev_idx = (self.selected_object_idx - 1) % len(self.objects)
                    self._select_object(prev_idx)

            # Number keys 1-9
            elif hasattr(event, 'key') and hasattr(gui.KeyName, 'ONE'):
                key_nums = {
                    gui.KeyName.ONE: 0, gui.KeyName.TWO: 1, gui.KeyName.THREE: 2,
                    gui.KeyName.FOUR: 3, gui.KeyName.FIVE: 4, gui.KeyName.SIX: 5,
                    gui.KeyName.SEVEN: 6, gui.KeyName.EIGHT: 7, gui.KeyName.NINE: 8
                }
                if event.key in key_nums:
                    idx = key_nums[event.key]
                    if idx < len(self.objects):
                        self._select_object(idx)

        return True

    def _select_object(self, obj_idx: int):
        """Select an object and update visualization"""
        print(f"\n=== Object Selected: #{obj_idx} ===")

        # Deselect previous
        if self.selected_object_idx is not None:
            # Reset previous bbox color
            old_bbox = self._create_bbox_lineset(
                self.objects[self.selected_object_idx]['bbox_min'],
                self.objects[self.selected_object_idx]['bbox_max'],
                [0.7, 0.7, 0.7]
            )
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.line_width = 2.0
            self.scene.scene.remove_geometry(f"bbox_{self.selected_object_idx}")
            self.scene.scene.add_geometry(f"bbox_{self.selected_object_idx}", old_bbox, mat)

            # Remove relationship lines
            self.scene.scene.remove_geometry("relationships")

        # Select new
        self.selected_object_idx = obj_idx
        obj = self.objects[obj_idx]

        # Highlight selected bbox with bright yellow and thick lines
        highlighted_bbox = self._create_bbox_lineset(
            obj['bbox_min'],
            obj['bbox_max'],
            [1.0, 0.843, 0.0]  # Bright gold/yellow
        )
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.line_width = 5.0  # Extra thick for selected object
        self.scene.scene.remove_geometry(f"bbox_{obj_idx}")
        self.scene.scene.add_geometry(f"bbox_{obj_idx}", highlighted_bbox, mat)

        # Update basic info
        info_text = f"Object #{obj['instance_id']}\n"
        info_text += f"Class: {obj['semantic_class'].upper()}\n"
        info_text += f"Points: {obj['num_points']:,}\n"
        info_text += f"Position: ({obj['centroid'][0]:.1f}, {obj['centroid'][1]:.1f}, {obj['centroid'][2]:.1f})"
        self.object_info_label.text = info_text

        # Generate and display contextual description
        contextual_desc = self._generate_contextual_description(obj_idx)
        self.description_label.text = contextual_desc

        # Compute and display relationships
        relationships = self._compute_object_relationships(obj_idx)
        if relationships:
            rel_text = ""
            for i, (rel_idx, rel_desc, rel_class) in enumerate(relationships[:5]):  # Show top 5
                rel_text += f"{i+1}. {rel_class}: {rel_desc}\n"
            self.relationships_label.text = rel_text

            # Draw lines to related objects
            self._draw_relationship_lines(obj_idx, [r[0] for r in relationships[:5]])
        else:
            self.relationships_label.text = "No nearby objects found"

        # Force GUI update
        self.window.post_redraw()

    def _draw_relationship_lines(self, source_idx: int, target_indices: List[int]):
        """Draw lines connecting selected object to related objects"""
        source_centroid = self.objects[source_idx]['centroid']

        points = [source_centroid]
        lines = []

        for i, target_idx in enumerate(target_indices):
            target_centroid = self.objects[target_idx]['centroid']
            points.append(target_centroid)
            lines.append([0, i + 1])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.5, 1.0, 0.0]] * len(lines))  # Bright lime green

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.line_width = 4.0  # Thicker relationship lines
        self.scene.scene.add_geometry("relationships", line_set, mat)
