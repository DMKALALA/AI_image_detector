"""
AI Image Detection Training Module

This module contains classes for training AI image detection models.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from PIL import Image
import io
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AIImageDataset(Dataset):
    """Custom dataset for AI image detection"""
    
    def __init__(self, images, labels, processor, transform=None):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                # Handle bytes or other formats
                image = Image.open(io.BytesIO(image)).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Process with the image processor
        try:
            inputs = self.processor(image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except Exception as e:
            logger.warning(f"Failed to process image {idx}: {e}")
            # Create a dummy tensor if processing fails
            pixel_values = torch.zeros(3, 224, 224)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DatasetProcessor:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self):
        self.dataset = None
        self.processed_images = []
        self.processed_labels = []
    
    def load_dataset(self, dataset_name="anonymous1233/twitter_AII"):
        """Load dataset from Hugging Face"""
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name)
            logger.info(f"Dataset loaded successfully. Train size: {len(self.dataset['train'])}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def prepare_data(self, max_samples=5000):
        """Prepare and preprocess the dataset"""
        if not self.dataset:
            logger.error("No dataset loaded")
            return [], []
        
        try:
            logger.info(f"Processing {max_samples} samples...")
            
            # Get the training split
            train_data = self.dataset['train']
            
            # Limit samples to prevent memory issues
            num_samples = min(max_samples, len(train_data))
            train_data = train_data.select(range(num_samples))
            
            images = []
            labels = []
            
            # Process images with progress bar
            for i, sample in enumerate(tqdm(train_data, desc="Processing images")):
                try:
                    # Get real image (twitter_image) - label 0
                    real_image = sample['twitter_image']
                    if real_image.mode != 'RGB':
                        real_image = real_image.convert('RGB')
                    real_image = real_image.resize((224, 224), Image.Resampling.LANCZOS)
                    images.append(real_image)
                    labels.append(0)  # Real image
                    
                    # Get AI-generated images - label 1
                    ai_image_keys = ['sd35_image', 'sd3_image', 'sd21_image', 'sdxl_image', 'dalle_image']
                    for ai_key in ai_image_keys:
                        if ai_key in sample and sample[ai_key] is not None:
                            ai_image = sample[ai_key]
                            if ai_image.mode != 'RGB':
                                ai_image = ai_image.convert('RGB')
                            ai_image = ai_image.resize((224, 224), Image.Resampling.LANCZOS)
                            images.append(ai_image)
                            labels.append(1)  # AI-generated image
                            break  # Only use one AI image per sample to balance the dataset
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {i}: {e}")
                    continue
            
            self.processed_images = images
            self.processed_labels = labels
            
            logger.info(f"Successfully processed {len(images)} images")
            return images, labels
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return [], []
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                images, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Second split: separate train and validation from remaining data
            val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
            )
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            return ([], []), ([], []), ([], [])

class AIImageDetector(nn.Module):
    """Custom model for AI image detection"""
    
    def __init__(self, model_name="microsoft/resnet-50", num_classes=2, dropout=0.3):
        super(AIImageDetector, self).__init__()
        
        # Load pre-trained model
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size
        if hasattr(self.backbone.config, 'hidden_size'):
            hidden_size = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'hidden_sizes'):
            hidden_size = self.backbone.config.hidden_sizes[-1]  # Use the last hidden size
        else:
            # For ResNet, use a default size
            hidden_size = 2048
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, pixel_values):
        # Get features from backbone
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Model trainer initialized on device: {self.device}")
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=1e-4, weight_decay=1e-5):
        """Train the model"""
        try:
            # Setup optimizer and loss function
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                
                # Training phase
                train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                
                # Store metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Clear cache to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(pixel_values)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=['Real', 'AI-Generated'])
        
        return accuracy, report, all_predictions, all_labels
    
    def save_model(self, model_path):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def plot_training_history(self, save_path):
        """Plot training history"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot losses
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracies
            ax2.plot(self.train_accuracies, label='Train Accuracy')
            ax2.plot(self.val_accuracies, label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot training history: {e}")
            raise
