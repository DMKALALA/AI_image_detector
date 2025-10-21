import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

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
        
        # Process image with the processor
        if self.processor:
            inputs = self.processor(image, return_tensors="pt")
            # Remove batch dimension
            for key in inputs:
                inputs[key] = inputs[key].squeeze(0)
        else:
            inputs = {'pixel_values': torch.tensor(np.array(image))}
        
        return inputs, torch.tensor(label, dtype=torch.long)

class AIImageDetector(nn.Module):
    """Custom model for AI image detection"""
    
    def __init__(self, model_name="microsoft/resnet-50", num_classes=2, dropout=0.3):
        super(AIImageDetector, self).__init__()
        
        # Load pre-trained model
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size
        hidden_size = self.backbone.config.hidden_size
        
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

class DatasetProcessor:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, dataset_name="anonymous1233/twitter_AII"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.processor = None
        
    def load_dataset(self):
        """Load the dataset from Hugging Face"""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset['train'])}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def prepare_data(self, max_samples=None):
        """Prepare data for training"""
        if not self.dataset:
            logger.error("Dataset not loaded")
            return None, None
        
        images = []
        labels = []
        
        # Process the dataset
        data = self.dataset['train']
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        logger.info(f"Processing {len(data)} samples...")
        
        for i, sample in enumerate(tqdm(data)):
            try:
                # Get real image (twitter_image) - label 0
                if 'twitter_image' in sample and sample['twitter_image']:
                    images.append(sample['twitter_image'])
                    labels.append(0)  # Real image
                
                # Get AI-generated images - label 1
                ai_models = ['sd35_image', 'sd3_image', 'sd21_image', 'sdxl_image', 'dalle_image']
                for model in ai_models:
                    if model in sample and sample[model]:
                        images.append(sample[model])
                        labels.append(1)  # AI-generated image
                        
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        logger.info(f"Prepared {len(images)} images: {labels.count(0)} real, {labels.count(1)} AI-generated")
        return images, labels
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            # Move to device
            pixel_values = inputs['pixel_values'].to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(pixel_values)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                pixel_values = inputs['pixel_values'].to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=1e-4):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        logger.info("Training completed!")
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                pixel_values = inputs['pixel_values'].to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                    target_names=['Real', 'AI-Generated'])
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return accuracy, report, all_predictions, all_labels
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    processor = DatasetProcessor()
    
    # Load dataset
    if not processor.load_dataset():
        return
    
    # Prepare data
    images, labels = processor.prepare_data(max_samples=5000)  # Limit for initial testing
    if not images:
        return
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(images, labels)
    
    # Initialize processor for images
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    
    # Create datasets
    train_dataset = AIImageDataset(X_train, y_train, image_processor)
    val_dataset = AIImageDataset(X_val, y_val, image_processor)
    test_dataset = AIImageDataset(X_test, y_test, image_processor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = AIImageDetector()
    trainer = ModelTrainer(model)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=5, learning_rate=1e-4)
    
    # Evaluate model
    accuracy, report, predictions, labels = trainer.evaluate(test_loader)
    
    # Save model
    trainer.save_model('trained_ai_detector.pth')
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
