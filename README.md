# SLAM Dataset Annotation Pipeline

Automated pipeline for semantic segmentation and natural language annotation of 3D point cloud data from SLAM-compatible datasets (KITTI Odometry).

## Overview

This project automates the process of:
1. **Downloading** KITTI Odometry dataset
2. **Preprocessing** 3D point clouds
3. **Segmentation** using deep learning models (PointNet, RandLA-Net, Cylinder3D, etc.)
4. **Object extraction** from segmented scenes
5. **NLP annotation** for semantic understanding (PointLLM, LL3DA, etc.)
6. **Visualization** of results

## Features

- **Automated Dataset Management**: Download and organize KITTI Odometry dataset
- **Modular Architecture**: Easy to swap segmentation and NLP models
- **Multiple Model Support**:
  - Segmentation: Simple PointNet, RandLA-Net, Cylinder3D, Panoptic-PolarNet, SalsaNext
  - NLP: Rule-based, PointLLM, LL3DA, MIT-SPARK
- **PyTorch Backend**: All models built with PyTorch for easy training/fine-tuning
- **Comprehensive Visualization**: 2D/3D point cloud visualization with semantic labels
- **Flexible Configuration**: YAML-based configuration system

## Project Structure

```
nlp_project/
├── configs/
│   └── default_config.yaml       # Configuration file
├── data/
│   └── kitti/                    # KITTI dataset (auto-downloaded)
├── models/
│   ├── __init__.py
│   ├── segmentation_model.py    # Segmentation model wrappers
│   └── nlp_model.py              # NLP model wrappers
├── utils/
│   ├── __init__.py
│   ├── point_cloud_utils.py     # Point cloud processing utilities
│   └── visualization.py          # Visualization tools
├── outputs/                      # Generated outputs
│   ├── annotations/              # JSON annotations
│   ├── objects/                  # Extracted objects
│   └── visualizations/           # Plots and figures
├── demo.py                       # Main pipeline script
├── download_kitti.py             # Dataset download script
├── quick_start.py                # Quick demo with synthetic data
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- ~50GB free space for KITTI dataset

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd nlp_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Quick Demo (No Download Required)

Run a quick demo with synthetic data to test the pipeline:

```bash
python quick_start.py
```

This will:
- Generate synthetic point cloud data
- Run segmentation and NLP models
- Display results and visualizations
- Save outputs to `outputs/quick_demo/`

### Option 2: Full Pipeline with KITTI Dataset

1. **Download KITTI dataset** (this will take a while):
   ```bash
   python download_kitti.py --components odometry_velodyne odometry_calib odometry_poses
   ```

2. **Run the complete pipeline**:
   ```bash
   python demo.py
   ```

3. **View results** in the `outputs/` directory:
   - `annotations/`: JSON files with object descriptions
   - `visualizations/`: 2D and 3D PNG images of segmented point clouds
   - `objects/`: Individual extracted objects as .npy files

   The pipeline will show:
   - **2D bird's eye view plots** (matplotlib)
   - **3D matplotlib plots** (saved as PNG)
   - **Interactive 3D viewer** (Open3D) - rotate, zoom, and explore!

## Configuration

Edit `configs/default_config.yaml` to customize the pipeline:

### Key Configuration Options

```yaml
# Which sequence to process
dataset:
  sequence: "00"  # KITTI sequences: 00-21

# Processing settings
processing:
  device: "cuda"        # Use "cpu" if no GPU
  num_scans: 5          # Number of scans to process (-1 for all)
  start_index: 0        # Starting scan

# Model selection
segmentation:
  model_type: "simple"  # Options: simple, randlanet, cylinder3d

nlp:
  model_type: "simple"  # Options: simple, pointllm, ll3da

# Visualization
visualization:
  enabled: true
  show_2d: true         # Bird's eye view plots
  show_3d: true         # Interactive Open3D viewer
  save_figures: true    # Save 2D and 3D matplotlib plots
```

## Usage Examples

### Download Specific Dataset Components

```bash
# Download only point clouds
python download_kitti.py --components odometry_velodyne

# Download everything
python download_kitti.py --components all
```

### Run Pipeline with Custom Config

```bash
python demo.py --config configs/my_config.yaml
```

### Process Specific Scans

Edit `configs/default_config.yaml`:
```yaml
processing:
  num_scans: 10       # Process 10 scans
  start_index: 50     # Start from scan 50
```

### Use Different Models

In `configs/default_config.yaml`:
```yaml
segmentation:
  model_type: "randlanet"  # Change to RandLA-Net
  checkpoint_path: "models/weights/randlanet.pth"

nlp:
  model_type: "pointllm"   # Change to PointLLM
  checkpoint_path: "models/weights/pointllm.pth"
```

## Advanced: Integrating Custom Models

### Adding a Segmentation Model

1. Create a new class in `models/segmentation_model.py`:

```python
class MySegmentationModel(BaseSegmentationModel):
    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        # Initialize your model

    def load_model(self, checkpoint_path):
        # Load your model weights

    def preprocess(self, points):
        # Preprocess point cloud

    def predict(self, points):
        # Run inference
        return labels
```

2. Update the factory function in `models/segmentation_model.py`:

```python
def create_segmentation_model(model_type, ...):
    if model_type == 'my_model':
        return MySegmentationModel(...)
```

3. Use in config:
```yaml
segmentation:
  model_type: "my_model"
```

### Adding an NLP Model

Follow a similar pattern in `models/nlp_model.py` by extending `BaseNLPModel`.

## Supported Models

### Segmentation Models

| Model | Status | Notes |
|-------|--------|-------|
| **Simple PointNet** | ✅ Implemented | Demo purposes, limited accuracy |
| **RandLA-Net** | 🔧 Wrapper only | Requires separate installation |
| **Cylinder3D** | 🔧 Wrapper only | Requires separate installation |
| **Panoptic-PolarNet** | ⏳ Coming soon | - |
| **SalsaNext** | ⏳ Coming soon | - |

### NLP Models

| Model | Status | Notes |
|-------|--------|-------|
| **Rule-based** | ✅ Implemented | Geometric feature-based descriptions |
| **PointLLM** | 🔧 Wrapper only | Requires separate installation |
| **LL3DA** | 🔧 Wrapper only | Requires separate installation |
| **MIT-SPARK** | ⏳ Coming soon | - |

## Output Format

### Annotation JSON Structure

```json
{
  "scan_id": 0,
  "num_points": 45231,
  "num_objects": 12,
  "scene_description": "Scene contains 12 detected objects...",
  "objects": [
    {
      "instance_id": 0,
      "semantic_label": 1,
      "num_points": 3421,
      "centroid": [5.2, -2.1, 0.3],
      "bbox_min": [3.1, -3.5, -1.0],
      "bbox_max": [7.3, -0.7, 1.6],
      "description": "A car (medium-sized) located ahead (nearby, ~5.5m)"
    }
  ]
}
```

## Dataset Information

### KITTI Odometry Dataset

- **Source**: [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- **Sequences**: 22 sequences (00-21)
- **Point Cloud Format**: Velodyne .bin files
- **Coordinate System**: LiDAR coordinate frame
- **Download Size**:
  - Velodyne: ~80 GB
  - Color Images: ~65 GB
  - Calibration: ~1 MB
  - Poses: ~3 MB

### KITTI Semantic Classes

The pipeline supports 20 semantic classes:
- Vehicle: car, bicycle, motorcycle, truck, other-vehicle
- Human: person, bicyclist, motorcyclist
- Ground: road, parking, sidewalk, other-ground, terrain
- Structure: building, fence, trunk, pole, traffic-sign
- Nature: vegetation

## Training Custom Models

### Segmentation Model Training

```python
from models.segmentation_model import SimpleSegmentationModel

# Initialize model
model = SimpleSegmentationModel(num_classes=20, device='cuda')

# Load your training data
# train_loader = ...

# Training loop
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        points, labels = batch
        logits = model.model(points)
        loss = criterion(logits.view(-1, 20), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save checkpoint
torch.save({
    'model_state_dict': model.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpoint.pth')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_scans` in config
   - Use CPU: set `device: "cpu"` in config
   - Increase voxel downsampling: set larger `voxel_size`

2. **Download Fails**
   - Check internet connection
   - Try downloading components one at a time
   - Use `--components` flag to download specific parts

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Visualization Window Doesn't Appear**
   - For headless systems, set `show_3d: false` in config
   - Check matplotlib backend

## Performance Tips

- **GPU**: Use CUDA-enabled GPU for 10-100x speedup
- **Batch Processing**: Increase batch size if GPU memory allows
- **Downsampling**: Adjust `voxel_size` to balance quality vs speed
- **Multiprocessing**: Process multiple scans in parallel (future feature)

## Citation

If you use this pipeline in your research, please cite the relevant datasets and models:

```bibtex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Implement more segmentation models (RandLA-Net, Cylinder3D, etc.)
- Add more sophisticated NLP models (PointLLM, LL3DA)
- Support for other datasets (nuScenes, Waymo, etc.)
- Multi-frame temporal consistency
- Fine-tuning scripts
- Model evaluation metrics

## License

This project is provided as-is for research and educational purposes.
Individual models and datasets may have their own licenses.

## Resources

- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **RandLA-Net**: https://github.com/QingyongHu/RandLA-Net
- **Cylinder3D**: https://github.com/xinge008/Cylinder3D
- **PointLLM**: https://github.com/OpenRobotLab/PointLLM
- **LL3DA**: https://github.com/Open3DA/LL3DA

## Contact

For questions or issues, please open an issue on the project repository.

---

**Happy Annotating! 🚗📊🤖**
