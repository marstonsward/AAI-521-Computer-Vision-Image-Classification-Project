"""
Visualization utilities for model evaluation and analysis.

Contains:
- plot_training_history: Plot loss and accuracy curves
- plot_confusion_matrix: Display confusion matrix
- compare_models: Side-by-side model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history (dict): Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title (str): Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Real', 'AI-Generated'], title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): Class name labels
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names=['Real', 'AI-Generated']):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): Class name labels
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def compare_models(results_dict, metric='accuracy'):
    """
    Create bar chart comparing multiple models.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        metric (str): Metric name for labels
    """
    models = list(results_dict.keys())
    values = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = plt.bar(models, values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_predictions(images, true_labels, pred_labels, class_names=['Real', 'AI-Generated'], n=8):
    """
    Visualize sample predictions.
    
    Args:
        images: Tensor of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names (list): Class name labels
        n (int): Number of images to display
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(n, len(images))):
        # Denormalize image
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get labels
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        
        # Plot
        axes[i].imshow(img)
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                         fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
