"""
NLP Model Wrappers for 3D Scene Understanding

Base classes and implementations for adding semantic descriptions to objects.
Supports integration with models like PointLLM, LL3DA, etc.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BaseNLPModel(ABC):
    """Base class for NLP models for 3D scene understanding"""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize NLP model

        Args:
            device: Device to run model on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")

        self.model = None

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load model weights"""
        pass

    @abstractmethod
    def describe_object(self, obj: Dict) -> str:
        """
        Generate textual description for an object

        Args:
            obj: Object dictionary with points, label, and metadata

        Returns:
            Text description of the object
        """
        pass

    @abstractmethod
    def describe_scene(self, objects: List[Dict]) -> str:
        """
        Generate textual description for entire scene

        Args:
            objects: List of object dictionaries

        Returns:
            Text description of the scene
        """
        pass

    def annotate_objects(self, objects: List[Dict]) -> List[Dict]:
        """
        Add text annotations to all objects

        Args:
            objects: List of object dictionaries

        Returns:
            List of objects with added 'description' field
        """
        annotated_objects = []

        for obj in objects:
            description = self.describe_object(obj)
            obj_annotated = obj.copy()
            obj_annotated['description'] = description
            annotated_objects.append(obj_annotated)

        return annotated_objects


class SimpleNLPModel(BaseNLPModel):
    """
    Simple rule-based NLP model for demonstration

    This provides basic object descriptions based on geometric features.
    Replace with PointLLM, LL3DA, etc. for advanced scene understanding.
    """

    # KITTI semantic class names
    CLASS_NAMES = {
        0: "unlabeled",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
    }

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load model (no model needed for rule-based approach)"""
        print("Using rule-based NLP model (no checkpoint needed)")

    def _compute_geometric_features(self, obj: Dict) -> Dict:
        """Compute geometric features for description"""
        points = obj['points'][:, :3]

        # Size
        bbox_size = obj['bbox_max'] - obj['bbox_min']
        volume = np.prod(bbox_size)

        # Shape features
        height = bbox_size[2]
        length = max(bbox_size[0], bbox_size[1])
        width = min(bbox_size[0], bbox_size[1])
        aspect_ratio = length / (width + 1e-8)

        # Position features
        centroid = obj['centroid']
        distance_from_origin = np.linalg.norm(centroid[:2])  # x-y distance
        height_from_ground = centroid[2]

        # Direction
        angle = np.arctan2(centroid[1], centroid[0])
        angle_deg = np.degrees(angle)

        # Determine cardinal direction
        if -45 <= angle_deg < 45:
            direction = "ahead"
        elif 45 <= angle_deg < 135:
            direction = "to the left"
        elif angle_deg >= 135 or angle_deg < -135:
            direction = "behind"
        else:
            direction = "to the right"

        return {
            'volume': volume,
            'height': height,
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'distance': distance_from_origin,
            'height_from_ground': height_from_ground,
            'direction': direction,
            'num_points': obj['num_points']
        }

    def describe_object(self, obj: Dict) -> str:
        """Generate textual description for an object"""
        # Get class name
        class_id = obj['semantic_label']
        class_name = self.CLASS_NAMES.get(class_id, f"object-{class_id}")

        # Compute features
        features = self._compute_geometric_features(obj)

        # Build description
        description_parts = []

        # Basic identification
        description_parts.append(f"A {class_name}")

        # Size description
        if class_id in [1, 4, 5]:  # vehicles
            if features['length'] > 5:
                size_desc = "large"
            elif features['length'] > 3:
                size_desc = "medium-sized"
            else:
                size_desc = "small"
            description_parts.append(f"({size_desc})")

        # Position
        description_parts.append(f"located {features['direction']}")

        # Distance
        if features['distance'] < 5:
            dist_desc = "very close"
        elif features['distance'] < 15:
            dist_desc = "nearby"
        elif features['distance'] < 30:
            dist_desc = "at moderate distance"
        else:
            dist_desc = "far away"

        description_parts.append(f"({dist_desc}, ~{features['distance']:.1f}m)")

        # Height information for certain objects
        if class_id in [13, 14, 16, 18]:  # buildings, fences, trunks, poles
            description_parts.append(f"with height of {features['height']:.1f}m")

        # Shape information
        if class_id in [1, 2, 3, 4, 5]:  # vehicles
            if features['aspect_ratio'] > 2:
                description_parts.append("(elongated shape)")
            elif features['aspect_ratio'] < 1.5:
                description_parts.append("(compact shape)")

        # Point density
        density = features['num_points'] / (features['volume'] + 1e-8)
        if density > 1000:
            description_parts.append("[high detail]")
        elif density < 100:
            description_parts.append("[sparse detail]")

        return " ".join(description_parts)

    def describe_scene(self, objects: List[Dict]) -> str:
        """Generate textual description for entire scene"""
        if not objects:
            return "Empty scene with no detected objects."

        # Count objects by class
        class_counts = {}
        for obj in objects:
            class_id = obj['semantic_label']
            class_name = self.CLASS_NAMES.get(class_id, f"object-{class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Build scene description
        description_parts = []

        description_parts.append(f"Scene contains {len(objects)} detected objects:")

        # Summarize by class
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            if count == 1:
                description_parts.append(f"- 1 {class_name}")
            else:
                description_parts.append(f"- {count} {class_name}s")

        # Spatial distribution
        centroids = np.array([obj['centroid'][:2] for obj in objects])
        if len(centroids) > 0:
            avg_distance = np.mean(np.linalg.norm(centroids, axis=1))
            description_parts.append(f"\nAverage object distance: {avg_distance:.1f}m")

            # Scene extent
            min_coords = np.min(centroids, axis=0)
            max_coords = np.max(centroids, axis=0)
            scene_extent = max_coords - min_coords
            description_parts.append(
                f"Scene extent: {scene_extent[0]:.1f}m x {scene_extent[1]:.1f}m"
            )

        return "\n".join(description_parts)


class PointLLMWrapper(BaseNLPModel):
    """
    Wrapper for PointLLM or similar vision-language models

    To use this:
    1. Install PointLLM: https://github.com/OpenRobotLab/PointLLM
    2. Download pretrained weights
    3. Update the implementation below
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        print("PointLLM wrapper - requires separate installation")
        print("See: https://github.com/OpenRobotLab/PointLLM")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load PointLLM model"""
        raise NotImplementedError(
            "PointLLM requires separate installation. "
            "Clone the repo and follow setup instructions."
        )

    def describe_object(self, obj: Dict) -> str:
        """Generate description using PointLLM"""
        raise NotImplementedError("Implement based on PointLLM API")

    def describe_scene(self, objects: List[Dict]) -> str:
        """Generate scene description using PointLLM"""
        raise NotImplementedError("Implement based on PointLLM API")


class LLaMA32NLPModel(BaseNLPModel):
    """
    LLaMA 3.2 1B model for intelligent object annotation

    Generates rich descriptions including:
    - Object identification and characteristics
    - Spatial relationships with other objects
    - Contextual scene understanding
    """

    CLASS_NAMES = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
        9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
        13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
        17: "terrain", 18: "pole", 19: "traffic-sign",
    }

    def __init__(self, device: str = 'cuda', use_quantization: bool = True):
        super().__init__(device)
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.all_objects = []  # Store all objects for spatial relationship computation

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load TinyLlama from HuggingFace (no auth required)"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Using TinyLlama - no authentication required and more reliable download
        # Alternative: "meta-llama/Llama-3.2-1B" (requires auth and network access)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print(f"Loading TinyLlama from {model_name}...")
        print("Note: First run will download model (~2GB). Subsequent runs use cache.")

        try:
            # Try loading from cache first (offline mode)
            try:
                print("Attempting to load from cache (offline mode)...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

                if self.use_quantization and self.device == 'cuda':
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        load_in_8bit=True,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                print("Loaded from cache successfully!")

            except Exception as cache_error:
                # Cache failed, try downloading
                print("Cache not found, downloading from HuggingFace...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                if self.use_quantization and self.device == 'cuda':
                    print("Using 8-bit quantization for efficient inference...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        load_in_8bit=True,
                        torch_dtype=torch.float16
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )

            # Configure tokenizer pad token to avoid warnings
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model.eval()
            print("TinyLlama loaded successfully!")

        except Exception as e:
            print(f"Error loading LLaMA model: {e}")
            print("\n" + "="*60)
            print("SOLUTION: Use TinyLlama (no authentication needed)")
            print("="*60)
            print("Edit models/nlp_model.py line 328:")
            print('  model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"')
            print("\nOr if you want to stick with LLaMA:")
            print("  1. Check your internet connection")
            print("  2. Try running: huggingface-cli login")
            print("  3. Download model manually if needed")
            print("="*60)
            print("\nFalling back to rule-based descriptions...")
            self.model = None

    def _compute_spatial_relationships(self, obj: Dict, all_objects: List[Dict]) -> List[str]:
        """
        Compute spatial relationships between this object and others

        Returns:
            List of spatial relationship descriptions
        """
        relationships = []
        obj_centroid = obj['centroid']
        obj_class = self.CLASS_NAMES.get(obj['semantic_label'], 'object')

        # Find nearby objects
        for other_obj in all_objects:
            if other_obj['instance_id'] == obj['instance_id']:
                continue

            other_centroid = other_obj['centroid']
            other_class = self.CLASS_NAMES.get(other_obj['semantic_label'], 'object')

            # Compute distance
            distance_3d = np.linalg.norm(obj_centroid - other_centroid)
            distance_2d = np.linalg.norm(obj_centroid[:2] - other_centroid[:2])

            # Only report close objects
            if distance_2d > 15:  # More than 15m away
                continue

            # Determine relative position
            dx = other_centroid[0] - obj_centroid[0]
            dy = other_centroid[1] - obj_centroid[1]
            dz = other_centroid[2] - obj_centroid[2]

            # Direction
            angle = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle)

            if -22.5 <= angle_deg < 22.5:
                direction = "ahead"
            elif 22.5 <= angle_deg < 67.5:
                direction = "ahead-left"
            elif 67.5 <= angle_deg < 112.5:
                direction = "to the left"
            elif 112.5 <= angle_deg < 157.5:
                direction = "behind-left"
            elif angle_deg >= 157.5 or angle_deg < -157.5:
                direction = "behind"
            elif -157.5 <= angle_deg < -112.5:
                direction = "behind-right"
            elif -112.5 <= angle_deg < -67.5:
                direction = "to the right"
            else:
                direction = "ahead-right"

            # Distance description
            if distance_2d < 2:
                dist_desc = "very close"
            elif distance_2d < 5:
                dist_desc = "close"
            elif distance_2d < 10:
                dist_desc = "nearby"
            else:
                dist_desc = f"{distance_2d:.1f}m away"

            # Relative height
            if abs(dz) < 0.5:
                height_rel = "at same level"
            elif dz > 0.5:
                height_rel = "above"
            else:
                height_rel = "below"

            # Create relationship description
            rel = f"{other_class} {direction} ({dist_desc}, {height_rel})"
            relationships.append(rel)

        # Return top 3 closest relationships
        return relationships[:3]

    def _generate_with_llama(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using LLaMA model"""
        if self.model is None:
            return "[LLaMA model not available]"

        try:
            # Format as chat
            messages = [
                {"role": "system", "content": "You are a helpful assistant that describes 3D objects and scenes concisely."},
                {"role": "user", "content": prompt}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(input_ids).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=0.6,  # Lower for more focused output
                    top_p=0.9,
                    top_k=50,  # Add top-k for better quality
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Reduce repetition
                )

            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            # Clean up response
            response = response.strip()
            # Remove incomplete sentences at the end
            if response and not response[-1] in '.!?':
                # Find last complete sentence
                last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_period > 0:
                    response = response[:last_period + 1]
            return response

        except Exception as e:
            print(f"Error generating with LLaMA: {e}")
            return "[Generation failed]"

    def describe_object(self, obj: Dict) -> str:
        """Generate intelligent description for an object"""
        # Get basic info
        class_id = obj['semantic_label']
        class_name = self.CLASS_NAMES.get(class_id, f"object-{class_id}")
        centroid = obj['centroid']
        bbox_size = obj['bbox_max'] - obj['bbox_min']
        num_points = obj['num_points']

        # Compute spatial relationships
        relationships = self._compute_spatial_relationships(obj, self.all_objects)

        # Build improved prompt for better descriptions
        distance = np.linalg.norm(centroid[:2])

        prompt = f"""You are analyzing a 3D LiDAR scan. Describe this object clearly and concisely.

Object: {class_name}
Distance from sensor: {distance:.1f}m
Dimensions: {bbox_size[0]:.1f}m wide x {bbox_size[1]:.1f}m deep x {bbox_size[2]:.1f}m tall"""

        if relationships:
            prompt += f"\nContext: {', '.join(relationships[:3])}"

        prompt += "\n\nProvide a brief, complete description in 2-3 sentences:"

        # Generate with LLaMA - increased tokens for complete sentences
        description = self._generate_with_llama(prompt, max_tokens=150)

        # If LLaMA fails, use rule-based fallback
        if not description or description == "[LLaMA model not available]":
            distance = np.linalg.norm(centroid[:2])
            description = f"A {class_name} located {distance:.1f}m away"
            if relationships:
                description += f". Near: {relationships[0]}"

        return description

    def describe_scene(self, objects: List[Dict]) -> str:
        """Generate intelligent scene description"""
        if not objects:
            return "Empty scene with no detected objects."

        # Store objects for spatial relationship computation
        self.all_objects = objects

        # Count by class
        class_counts = {}
        for obj in objects:
            class_id = obj['semantic_label']
            class_name = self.CLASS_NAMES.get(class_id, f"object-{class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Summarize
        summary_parts = []
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            summary_parts.append(f"{count} {class_name}{'s' if count > 1 else ''}")

        summary = ", ".join(summary_parts)

        # Build prompt for LLaMA
        prompt = f"""Describe this 3D LIDAR scene in 2-3 sentences:

Total objects: {len(objects)}
Objects detected: {summary}

Describe what the scene likely represents (e.g., urban street, parking lot, etc.) and note any interesting spatial arrangements."""

        # Generate with LLaMA
        scene_description = self._generate_with_llama(prompt, max_tokens=150)

        # Fallback
        if not scene_description or scene_description == "[LLaMA model not available]":
            scene_description = f"Scene contains {len(objects)} objects: {summary}"

        return scene_description


class LL3DAWrapper(BaseNLPModel):
    """
    Wrapper for LL3DA (Language-driven Large-scale 3D Scene Understanding)

    To use this:
    1. Install LL3DA: https://github.com/Open3DA/LL3DA
    2. Download pretrained weights
    3. Update the implementation below
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        print("LL3DA wrapper - requires separate installation")
        print("See: https://github.com/Open3DA/LL3DA")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load LL3DA model"""
        raise NotImplementedError(
            "LL3DA requires separate installation. "
            "Clone the repo and follow setup instructions."
        )

    def describe_object(self, obj: Dict) -> str:
        """Generate description using LL3DA"""
        raise NotImplementedError("Implement based on LL3DA API")

    def describe_scene(self, objects: List[Dict]) -> str:
        """Generate scene description using LL3DA"""
        raise NotImplementedError("Implement based on LL3DA API")


# Factory function for easy model creation
def create_nlp_model(model_type: str = 'llama',
                     device: str = 'cuda',
                     checkpoint_path: Optional[str] = None) -> BaseNLPModel:
    """
    Factory function to create NLP models

    Args:
        model_type: Type of model ('llama', 'simple', 'pointllm', 'll3da')
        device: Device to run on
        checkpoint_path: Path to model checkpoint

    Returns:
        NLP model instance
    """
    if model_type == 'llama':
        model = LLaMA32NLPModel(device=device)
    elif model_type == 'simple':
        model = SimpleNLPModel(device=device)
    elif model_type == 'pointllm':
        model = PointLLMWrapper(device=device)
    elif model_type == 'll3da':
        model = LL3DAWrapper(device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_model(checkpoint_path)
    return model
