# SalsaNext Integration Guide

## Overview

This directory is prepared for SalsaNext model integration. The wrapper in `segmentation_model.py` is ready to use SalsaNext for semantic segmentation.

## Setup Instructions

### Option 1: Copy SalsaNext Files (Recommended)

1. Clone the official SalsaNext repository:
   ```bash
   git clone https://github.com/TiagoCortinhal/SalsaNext.git
   ```

2. Copy the model files:
   ```bash
   # Copy the architecture files
   cp SalsaNext/train/tasks/semantic/modules/SalsaNext.py models/salsanext/model.py
   cp SalsaNext/train/tasks/semantic/modules/SalsaNextEncoder.py models/salsanext/
   cp SalsaNext/train/tasks/semantic/modules/SalsaNextDecoder.py models/salsanext/
   ```

3. Update imports in `model.py` to match the new structure

### Option 2: Install as Package

If SalsaNext is available as a pip package:
```bash
pip install salsanext
```

Then update `models/salsanext/model.py`:
```python
from salsanext import SalsaNext
```

## Using Your Trained Weights

Once you have trained SalsaNext weights:

1. Place your checkpoint file in `models/weights/`:
   ```
   models/weights/salsanext_kitti.pth
   ```

2. Update the config file `configs/default_config.yaml`:
   ```yaml
   segmentation:
     model_type: "salsanext"
     num_classes: 20
     checkpoint_path: "models/weights/salsanext_kitti.pth"
   ```

3. Run the pipeline:
   ```bash
   python demo.py
   ```

## Model Architecture

SalsaNext uses:
- **Input**: Range images (64 x 2048) with 5 channels [range, x, y, z, intensity]
- **Architecture**: ResNet-based encoder-decoder with skip connections
- **Output**: Per-pixel semantic predictions

## Checkpoint Format

The wrapper supports multiple checkpoint formats:
- `checkpoint['model_state_dict']` (PyTorch Lightning)
- `checkpoint['state_dict']` (Standard PyTorch)
- Direct state dict

## FOV Parameters

Default KITTI parameters (can be adjusted in `SalsaNextWrapper.__init__`):
- Height: 64 pixels
- Width: 2048 pixels
- FOV up: 3.0°
- FOV down: -25.0°

For different sensors, modify these values in `models/segmentation_model.py`:
```python
self.proj_H = 64  # Your sensor height
self.proj_W = 2048  # Your sensor width
self.proj_fov_up = 3.0  # Your FOV up
self.proj_fov_down = -25.0  # Your FOV down
```

## Troubleshooting

### Import Errors
If you get `ImportError: cannot import name 'SalsaNext'`:
- Verify the model files are in `models/salsanext/`
- Check that `__init__.py` exists
- Ensure the class name matches in `model.py`

### Checkpoint Loading Errors
If checkpoint loading fails:
- Check the file path is correct
- Verify the checkpoint was trained with the same number of classes
- Try loading the checkpoint manually to inspect its structure:
  ```python
  import torch
  ckpt = torch.load('path/to/checkpoint.pth')
  print(ckpt.keys())
  ```

### Shape Mismatches
If you get shape errors:
- Verify your point cloud has 4 channels: [x, y, z, intensity]
- Check FOV parameters match your sensor
- Ensure num_classes matches your training

## References

- **Paper**: [SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds](https://arxiv.org/abs/2003.03653)
- **Code**: https://github.com/TiagoCortinhal/SalsaNext
- **Dataset**: Semantic KITTI (http://semantic-kitti.org/)
