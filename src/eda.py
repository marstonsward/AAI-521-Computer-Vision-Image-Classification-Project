"""
Exploratory Data Analysis (EDA) utilities.

Contains functions for visualizing and analyzing the dataset before training.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def analyze_class_distribution(labels, class_names=['Real', 'AI-Generated']):
    """
    Analyze and print class distribution statistics.
    
    Args:
        labels (list): List of labels
        class_names (list): Names of classes (default: ['Real', 'AI-Generated'])
    
    Returns:
        dict: Dictionary with class counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    stats = {}
    print("📊 Class Distribution:")
    for label, count in zip(unique, counts):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = (count / total) * 100
        stats[class_name] = {'count': count, 'percentage': percentage}
        print(f"   {class_name}: {count:,} images ({percentage:.1f}%)")
    
    return stats


def plot_class_distribution(labels, class_names=['Real', 'AI-Generated'], 
                           title='Class Distribution', figsize=(8, 5)):
    """
    Plot class distribution as a bar chart.
    
    Args:
        labels (list): List of labels
        class_names (list): Names of classes
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(unique)]
    bars = plt.bar(class_names[:len(unique)], counts, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualize_samples(images, labels, class_names=['Real', 'AI-Generated'],
                     n_per_class=4, figsize=(16, 8), seed=None):
    """
    Display sample images from each class in a grid.
    
    Args:
        images (list): List of PIL images
        labels (list): List of corresponding labels
        class_names (list): Names of classes
        n_per_class (int): Number of samples to show per class
        figsize (tuple): Figure size (width, height)
        seed (int, optional): Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Separate images by class
    classes = {}
    for img, lbl in zip(images, labels):
        if lbl not in classes:
            classes[lbl] = []
        classes[lbl].append(img)
    
    # Create subplot grid
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=figsize)
    
    # Handle case where axes might not be 2D
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    elif n_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot samples for each class
    for class_idx, (label, class_images) in enumerate(sorted(classes.items())):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        color = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][label % 4]
        
        for img_idx in range(n_per_class):
            if img_idx < len(class_images):
                sample_img = random.choice(class_images)
                axes[class_idx, img_idx].imshow(sample_img)
            else:
                axes[class_idx, img_idx].axis('off')
            
            if img_idx == 0:
                axes[class_idx, img_idx].set_ylabel(class_name, 
                                                    fontsize=12, 
                                                    fontweight='bold',
                                                    color=color)
            
            axes[class_idx, img_idx].set_title(class_name if img_idx == n_per_class // 2 else '',
                                               fontsize=12, fontweight='bold', color=color)
            axes[class_idx, img_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_batch(dataloader, class_names=['Real', 'AI-Generated'],
                   n_images=8, denormalize=True, 
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   title='Sample Batch'):
    """
    Visualize a batch of images from a DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names (list): Names of classes
        n_images (int): Number of images to display
        denormalize (bool): Whether to denormalize images
        mean (list): Mean values for denormalization
        std (list): Std values for denormalization
        title (str): Title for the plot
    """
    # Get one batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Denormalize if requested
    if denormalize:
        mean_array = np.array(mean)
        std_array = np.array(std)
    
    # Plot images
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i in range(n_images):
        if i < len(images):
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            
            if denormalize:
                img = std_array * img + mean_array
                img = np.clip(img, 0, 1)
            
            label = labels[i].item()
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            color = '#e74c3c' if label == 1 else '#3498db'
            
            axes[i].imshow(img)
            axes[i].set_title(class_name, fontsize=12, fontweight='bold', color=color)
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle(title, 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def dataset_summary(train_images, train_labels, val_images, val_labels, 
                   test_images, test_labels, class_names=['Real', 'AI-Generated']):
    """
    Print comprehensive dataset summary statistics.
    
    Args:
        train_images, train_labels: Training data
        val_images, val_labels: Validation data
        test_images, test_labels: Test data
        class_names (list): Names of classes
    """
    total_size = len(train_images) + len(val_images) + len(test_images)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"\nTotal Images: {total_size:,}")
    print(f"\nSplit:")
    print(f"   Training:   {len(train_images):,} images ({len(train_images)/total_size*100:.1f}%)")
    print(f"   Validation: {len(val_images):,} images ({len(val_images)/total_size*100:.1f}%)")
    print(f"   Test:       {len(test_images):,} images ({len(test_images)/total_size*100:.1f}%)")
    
    # Training set distribution
    print(f"\nTraining Set Distribution:")
    train_counter = Counter(train_labels)
    for label, count in sorted(train_counter.items()):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = (count / len(train_labels)) * 100
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    # Validation set distribution
    print(f"\nValidation Set Distribution:")
    val_counter = Counter(val_labels)
    for label, count in sorted(val_counter.items()):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = (count / len(val_labels)) * 100
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    # Test set distribution
    print(f"\nTest Set Distribution:")
    test_counter = Counter(test_labels)
    for label, count in sorted(test_counter.items()):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = (count / len(test_labels)) * 100
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    print("="*60)


def quick_eda(images, labels, class_names=['Real', 'AI-Generated']):
    """
    Perform quick EDA with both statistics and visualizations.
    
    Args:
        images (list): List of images
        labels (list): List of labels
        class_names (list): Names of classes
    """
    print("\n🔍 Quick EDA\n")
    
    # Statistics
    stats = analyze_class_distribution(labels, class_names)
    
    # Visualizations
    print("\n📊 Plotting class distribution...")
    plot_class_distribution(labels, class_names)
    
    print("\n🖼️  Displaying sample images...")
    visualize_samples(images, labels, class_names)
    
    return stats
