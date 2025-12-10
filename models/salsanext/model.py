"""
SalsaNext Model Implementation

Imported from the official SalsaNext repository:
https://github.com/TiagoCortinhal/SalsaNext

This is a real-time semantic segmentation model for LiDAR point clouds
using range image representation.
"""

# Import the actual SalsaNext model
from .SalsaNext import *

# Re-export for easier imports
__all__ = ['SalsaNext']
