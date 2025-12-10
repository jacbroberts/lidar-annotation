"""Model wrappers for segmentation and NLP tasks"""

from .segmentation_model import BaseSegmentationModel, SimpleSegmentationModel
from .nlp_model import BaseNLPModel, SimpleNLPModel

__all__ = [
    'BaseSegmentationModel',
    'SimpleSegmentationModel',
    'BaseNLPModel',
    'SimpleNLPModel'
]
