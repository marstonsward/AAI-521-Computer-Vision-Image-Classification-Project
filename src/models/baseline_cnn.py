"""
Baseline CNN architecture for AI-generated image detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture for binary classification of AI-generated vs real images.
    
    This model implements a simple but effective CNN with:
    - Multiple convolutional blocks with batch normalization and dropout
    - Progressive feature map reduction
    - Global average pooling to reduce parameters
    - Binary classification head
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 input_channels: int = 3,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True):
        """
        Initialize the baseline CNN.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Feature extraction layers
        self.features = self._make_feature_layers()
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_feature_layers(self) -> nn.Sequential:
        """
        Create the feature extraction layers.
        
        Returns:
            Sequential container with convolutional layers
        """
        layers = []
        
        # Block 1: 3 -> 64
        layers.extend(self._make_conv_block(3, 64, kernel_size=7, stride=2, padding=3))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Block 2: 64 -> 128
        layers.extend(self._make_conv_block(64, 128))
        layers.extend(self._make_conv_block(128, 128))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 3: 128 -> 256
        layers.extend(self._make_conv_block(128, 256))
        layers.extend(self._make_conv_block(256, 256))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 4: 256 -> 512
        layers.extend(self._make_conv_block(256, 512))
        layers.extend(self._make_conv_block(512, 512))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    def _make_conv_block(self, 
                        in_channels: int, 
                        out_channels: int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1) -> list:
        """
        Create a convolutional block with optional batch norm and activation.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            
        Returns:
            List of layers for the conv block
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not self.use_batch_norm)
        ]
        
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from (if None, returns final features)
            
        Returns:
            Feature maps tensor
        """
        if layer_name is None:
            # Return features before global average pooling
            return self.features(x)
        else:
            # Implementation for specific layer extraction could be added here
            raise NotImplementedError("Specific layer extraction not implemented")
    
    def count_parameters(self) -> tuple:
        """
        Count total and trainable parameters.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    def get_model_info(self) -> dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model statistics
        """
        total_params, trainable_params = self.count_parameters()
        
        return {
            'model_name': 'BaselineCNN',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class BaselineCNNSmall(BaselineCNN):
    """
    Smaller version of the baseline CNN for faster training and inference.
    """
    
    def _make_feature_layers(self) -> nn.Sequential:
        """Create smaller feature extraction layers."""
        layers = []
        
        # Block 1: 3 -> 32
        layers.extend(self._make_conv_block(3, 32, kernel_size=5, stride=2, padding=2))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 2: 32 -> 64
        layers.extend(self._make_conv_block(32, 64))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 3: 64 -> 128
        layers.extend(self._make_conv_block(64, 128))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 4: 128 -> 256
        layers.extend(self._make_conv_block(128, 256))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    def __init__(self, **kwargs):
        # Override classifier for smaller model
        super().__init__(**kwargs)
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(128, self.num_classes)
        )