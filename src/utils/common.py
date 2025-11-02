"""
Common utility functions for the AI-generated image detection project.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    Optimized for Mac M4 and Google Colab compatibility.
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon M4)")
        logger.info("MPS allows GPU acceleration on Mac M4")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        logger.warning("For better performance, consider using CUDA (Colab) or MPS (Mac M4)")
    
    return device


def save_model(model: nn.Module, 
               filepath: str, 
               metadata: Optional[Dict[str, Any]] = None,
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None):
    """
    Save model with metadata and optional training state.
    
    Args:
        model: PyTorch model to save
        filepath: Path where to save the model
        metadata: Optional metadata dictionary
        optimizer: Optional optimizer state to save
        epoch: Optional current epoch number
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    
    # Add optional components
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    # Save model
    torch.save(save_dict, filepath)
    
    # Save human-readable metadata
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved to {filepath}")


def load_model(model: nn.Module, 
               filepath: str, 
               device: Optional[torch.device] = None,
               strict: bool = True) -> Dict[str, Any]:
    """
    Load model from file.
    
    Args:
        model: Model instance to load weights into
        filepath: Path to saved model file
        device: Device to load model on
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary containing loaded metadata and other info
    """
    if device is None:
        device = get_device()
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model.to(device)
    
    logger.info(f"Model loaded from {filepath}")
    
    # Return metadata and other info
    return {
        'metadata': checkpoint.get('metadata', {}),
        'epoch': checkpoint.get('epoch'),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'model_class': checkpoint.get('model_class')
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_count = sum(p.numel() for p in model.parameters())
    # Assume float32 (4 bytes per parameter)
    size_mb = param_count * 4 / (1024 * 1024)
    return size_mb


def create_directory_structure(base_dir: str):
    """
    Create the standard directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/splits',
        'models',
        'results/figures',
        'results/metrics', 
        'results/reports',
        'logs'
    ]
    
    base_path = Path(base_dir)
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file for empty directories
        gitkeep_path = dir_path / '.gitkeep'
        if not any(dir_path.iterdir()):  # Directory is empty
            gitkeep_path.touch()
    
    logger.info(f"Directory structure created in {base_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level {log_level}")
    return root_logger


class EarlyStopping:
    """
    Early stopping callback to stop training when a metric stops improving.
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.is_better = self._get_comparison_function()
    
    def _get_comparison_function(self):
        """Get the appropriate comparison function based on mode."""
        if self.mode == 'max':
            return lambda current, best: current > best + self.min_delta
        else:
            return lambda current, best: current < best - self.min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score to evaluate
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count