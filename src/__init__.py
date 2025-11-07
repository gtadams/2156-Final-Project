"""
STP Classifier Package
Machine learning pipeline for classifying images based on 3D CAD models.
"""

from .stp_loader import STPLoader
from .renderer import Renderer
from .trainer import STAPClassifier, STPImageDataset
from .classifier import ImageClassifier

__version__ = '1.0.0'

__all__ = [
    'STPLoader',
    'Renderer',
    'STAPClassifier',
    'STPImageDataset',
    'ImageClassifier',
]
