"""
Training and evaluation utilities.

Contains:
- Trainer: Main training class with AMP support
- evaluate_model: Model evaluation function
- save_checkpoint/load_checkpoint: Model persistence
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class Trainer:
    """
    Training manager with automatic mixed precision (AMP) support.
    
    Args:
        model (torch.nn.Module): Model to train
        device (torch.device): Device to train on
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        use_amp (bool): Use automatic mixed precision (default: True)
    """
    def __init__(self, model, device, criterion, optimizer, use_amp=True):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_amp = use_amp
        
        # Initialize AMP scaler if using CUDA
        if use_amp and device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_wts = None
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    # Handle BCE loss: convert labels to float and match output shape
                    if outputs.dim() == 2 and outputs.size(1) == 1:
                        labels_for_loss = labels.float().unsqueeze(1)
                    else:
                        labels_for_loss = labels
                    loss = self.criterion(outputs, labels_for_loss)
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                # Handle BCE loss: convert labels to float and match output shape
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    labels_for_loss = labels.float().unsqueeze(1)
                else:
                    labels_for_loss = labels
                loss = self.criterion(outputs, labels_for_loss)
                loss.backward()
                self.optimizer.step()
            
            # Calculate accuracy
            batch_size = labels.size(0)
            
            # Handle both binary (BCEWithLogitsLoss) and multi-class (CrossEntropyLoss)
            if outputs.dim() == 1 or outputs.size(1) == 1:
                # Binary classification: apply sigmoid and threshold
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            else:
                # Multi-class: use argmax
                _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total += batch_size
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """Validate model on validation set."""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                # Handle BCE loss: convert labels to float and match output shape
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    labels_for_loss = labels.float().unsqueeze(1)
                else:
                    labels_for_loss = labels
                loss = self.criterion(outputs, labels_for_loss)
                
                batch_size = labels.size(0)
                
                # Handle both binary (BCEWithLogitsLoss) and multi-class (CrossEntropyLoss)
                if outputs.dim() == 1 or outputs.size(1) == 1:
                    # Binary classification: apply sigmoid and threshold
                    preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                else:
                    # Multi-class: use argmax
                    _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total += batch_size
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader, val_loader, num_epochs, save_path=None, verbose=True):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs (int): Number of epochs to train
            save_path (str, optional): Path to save best model
            verbose (bool): Print training progress
        
        Returns:
            dict: Training history
        """
        start_time = time.time()
        
        for epoch in range(num_epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                
                if save_path:
                    save_checkpoint(self.model, save_path)
                    if verbose:
                        print(f"💾 Saved best model (val_acc: {val_acc:.4f})")
        
        # Load best weights
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n✅ Training complete in {elapsed:.0f}s")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader: DataLoader for evaluation
        device (torch.device): Device to evaluate on
    
    Returns:
        tuple: (accuracy, all_preds, all_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy, all_preds, all_labels


def save_checkpoint(model, path):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
