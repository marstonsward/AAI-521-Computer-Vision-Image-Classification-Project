"""
Truth in Pixels - AI-Generated Image Detection
Source code package for the computer vision project.
"""

__version__ = "1.0.0"
__author__ = "AAI-521 Team"
__email__ = "team@example.com"

# Import key modules for easy access
from .data import DatasetManager, ImageTransforms
from .models import BaselineCNN, MobileNetV2Classifier
from .utils import set_seed, save_model, load_model
from .evaluation import ModelEvaluator, MetricsCalculator

__all__ = [
    "DatasetManager",
    "ImageTransforms", 
    "BaselineCNN",
    "MobileNetV2Classifier",
    "ModelEvaluator",
    "MetricsCalculator",
    "set_seed",
    "save_model", 
    "load_model"
]