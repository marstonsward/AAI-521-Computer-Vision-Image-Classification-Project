"""
Model architectures for AI-generated image detection.
"""

from .baseline_cnn import BaselineCNN
from .mobilenet_classifier import MobileNetV2Classifier
from .trainer import ModelTrainer

__all__ = ["BaselineCNN", "MobileNetV2Classifier", "ModelTrainer"]