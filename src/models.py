"""
Model architectures for AI-generated image detection.

Contains:
- CustomCNN: Baseline CNN with 3 conv layers
- get_resnet50: ResNet50 with transfer learning
- get_efficientnet_b2: EfficientNet-B2 with transfer learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for binary image classification.
    
    Architecture:
        - 3 convolutional layers (32, 64, 128 filters)
        - Batch normalization after each conv layer
        - MaxPooling after each conv block
        - Dropout for regularization
        - 2 fully connected layers
    
    Args:
        num_classes (int): Number of output classes (default: 1 for binary)
    """
    def __init__(self, num_classes=1):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        # After 3 pooling layers: 224/8 = 28, so 128 * 28 * 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def get_resnet50(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Get ResNet50 model with custom classifier for transfer learning.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze all layers except classifier (default: True)
    
    Returns:
        torch.nn.Module: ResNet50 model
    """
    # Load pretrained ResNet50
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def get_efficientnet_b2(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Get EfficientNet-B2 model with custom classifier for transfer learning.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze all layers except classifier (default: True)
    
    Returns:
        torch.nn.Module: EfficientNet-B2 model
    """
    # Load pretrained EfficientNet-B2
    if pretrained:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b2(weights=None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def count_parameters(model):
    """
    Count trainable and total parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
