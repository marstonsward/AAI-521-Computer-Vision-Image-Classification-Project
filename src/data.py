"""
Data loading and preprocessing utilities.

Contains:
- HuggingFaceImageDataset: Custom PyTorch Dataset
- get_transforms: Create train/val data transforms
- prepare_data: Load and split dataset
"""

import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class HuggingFaceImageDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for Hugging Face images.
    
    Args:
        images (list): List of PIL images
        labels (list): List of corresponding labels (0=Real, 1=AI-Generated)
        transform (callable, optional): Optional transform to apply to images
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augment=True):
    """
    Get image transformation pipelines.
    
    Args:
        img_size (int): Target image size (default: 224)
        augment (bool): Apply data augmentation (for training)
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    # ImageNet normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def prepare_data(dataset_name="Hemg/AI-Generated-vs-Real-Images-Datasets",
                 train_ratio=0.8,
                 val_ratio=0.1,
                 seed=42):
    """
    Load and split Hugging Face dataset.
    
    Args:
        dataset_name (str): Hugging Face dataset identifier
        train_ratio (float): Proportion for training (default: 0.8)
        val_ratio (float): Proportion for validation (default: 0.1)
        seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Extract images and labels
    images = [x['image'] for x in dataset['train']]
    labels = [x['label'] for x in dataset['train']]
    
    # Shuffle with seed
    random.seed(seed)
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    
    # Calculate split indices
    total_size = len(images)
    train_end = int(train_ratio * total_size)
    val_end = int((train_ratio + val_ratio) * total_size)
    
    # Split data
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]
    
    return (train_images, train_labels,
            val_images, val_labels,
            test_images, test_labels)


def create_dataloaders(train_images, train_labels,
                      val_images, val_labels,
                      test_images, test_labels,
                      batch_size=32,
                      num_workers=2):
    """
    Create PyTorch DataLoaders from image/label lists.
    
    Args:
        train_images, train_labels: Training data
        val_images, val_labels: Validation data
        test_images, test_labels: Test data
        batch_size (int): Batch size for dataloaders (default: 32)
        num_workers (int): Number of worker processes (default: 2)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # Create datasets
    train_dataset = HuggingFaceImageDataset(train_images, train_labels, train_transform)
    val_dataset = HuggingFaceImageDataset(val_images, val_labels, val_transform)
    test_dataset = HuggingFaceImageDataset(test_images, test_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
