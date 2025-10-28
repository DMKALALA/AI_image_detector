"""
Robust AI Image Detection Training System
Uses multiple datasets and ensemble architecture for better generalization
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from efficientnet_pytorch import EfficientNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustDatasetProcessor:
    """Process multiple datasets for robust training"""
    
    def __init__(self):
        self.datasets = {}
        self.processed_data = {'images': [], 'labels': [], 'sources': []}
        
    def load_multiple_datasets(self):
        """Load GenImage datasets for robust training"""
        datasets_to_load = [
            {
                'name': 'Hemg/AI-Generated-vs-Real-Images-Datasets',
                'weight': 0.5,
                'max_samples': 15000
            },
            {
                'name': 'cifake',
                'weight': 0.5,
                'max_samples': 15000
            }
        ]
        
        for dataset_config in datasets_to_load:
            try:
                logger.info(f"Loading dataset: {dataset_config['name']}")
                dataset = load_dataset(dataset_config['name'])
                self.datasets[dataset_config['name']] = dataset
                
                # Process dataset
                images, labels = self._process_dataset(
                    dataset_config['name'], 
                    dataset, 
                    dataset_config['max_samples']
                )
                
                # Add to processed data
                self.processed_data['images'].extend(images)
                self.processed_data['labels'].extend(labels)
                self.processed_data['sources'].extend([dataset_config['name']] * len(images))
                
                logger.info(f"Processed {len(images)} samples from {dataset_config['name']}")
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_config['name']}: {e}")
                continue
                
        logger.info(f"Total processed samples: {len(self.processed_data['images'])}")
        return self.processed_data
    
    def _process_dataset(self, dataset_name, dataset, max_samples):
        """Process a single dataset and extract images/labels"""
        images = []
        labels = []
        
        train_data = dataset['train']
        num_samples = min(max_samples, len(train_data))
        train_data = train_data.select(range(num_samples))
        
        for i, sample in enumerate(tqdm(train_data, desc=f"Processing {dataset_name}")):
            try:
                if dataset_name == 'Hemg/AI-Generated-vs-Real-Images-Datasets':
                    # Process Hemg dataset
                    if 'image' in sample:
                        image = sample['image']
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Resize to standard size
                        image = image.resize((224, 224))
                        images.append(np.array(image))
                        
                        # Label: 0 for real, 1 for AI
                        if 'label' in sample:
                            labels.append(sample['label'])
                        else:
                            # Default to real if no label
                            labels.append(0)
                            
                elif dataset_name == 'cifake':
                    # Process CIFake dataset
                    if 'image' in sample:
                        image = sample['image']
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Resize to standard size
                        image = image.resize((224, 224))
                        images.append(np.array(image))
                        
                        # Label: 0 for real, 1 for AI
                        if 'label' in sample:
                            labels.append(sample['label'])
                        else:
                            # Default to real if no label
                            labels.append(0)
                            
            except Exception as e:
                logger.warning(f"Failed to process sample {i} from {dataset_name}: {e}")
                continue
                
        return images, labels

class RobustAIImageDetector(nn.Module):
    """Ensemble AI Image Detector with multiple backbones"""
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(RobustAIImageDetector, self).__init__()
        
        # ResNet50 backbone
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        
        # EfficientNet-B0 backbone
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        
        # Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        
        # Ensemble fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Get predictions from each backbone
        resnet_out = self.resnet(x)
        efficientnet_out = self.efficientnet(x)
        vit_out = self.vit(x)
        
        # Concatenate outputs
        combined = torch.cat([resnet_out, efficientnet_out, vit_out], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        return output

class RobustModelTrainer:
    """Advanced trainer with cross-validation and robust techniques"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        if scheduler:
            scheduler.step()
            
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=25, learning_rate=5e-5, weight_decay=1e-4):
        """Train the model with advanced techniques"""
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'robust_ai_detector_best.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return best_val_acc

def create_robust_training_config():
    """Create robust training configuration for GenImage datasets"""
    config = {
        "datasets": {
            "hemg": {
                "name": "Hemg/AI-Generated-vs-Real-Images-Datasets",
                "weight": 0.5,
                "max_samples": 15000,
                "description": "High-quality AI vs Real image dataset"
            },
            "cifake": {
                "name": "cifake",
                "weight": 0.5,
                "max_samples": 15000,
                "description": "CIFAR-based fake detection dataset"
            }
        },
        "model": {
            "architecture": "ensemble",
            "backbones": ["resnet50", "efficientnet_b0", "vit_b_16"],
            "dropout": 0.5,
            "num_classes": 2
        },
        "training": {
            "epochs": 25,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "gradient_clipping": 1.0
        },
        "data_augmentation": {
            "enabled": True,
            "transforms": [
                {"name": "HorizontalFlip", "p": 0.5},
                {"name": "ShiftScaleRotate", "shift_limit": 0.0625, "scale_limit": 0.1, "rotate_limit": 15, "p": 0.5},
                {"name": "RandomBrightnessContrast", "brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
                {"name": "GaussNoise", "p": 0.2},
                {"name": "Cutout", "num_holes": 8, "max_h_size": 8, "max_w_size": 8, "p": 0.2}
            ]
        },
        "paths": {
            "model_save_path": "robust_ai_detector.pth",
            "plots_save_path": "robust_training_plots/",
            "logs_save_path": "robust_training_logs/"
        }
    }
    return config

def plot_training_results(trainer, save_path="robust_training_plots/"):
    """Plot training results"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.plot(trainer.val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(trainer.train_accuracies, label='Train Accuracy')
    ax2.plot(trainer.val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_results.png'))
    plt.close()
    
    logger.info(f"Training plots saved to {save_path}")

def main():
    """Main training function"""
    logger.info("Starting robust AI image detection training")
    
    # Load configuration
    config = create_robust_training_config()
    
    # Process datasets
    processor = RobustDatasetProcessor()
    data = processor.load_multiple_datasets()
    
    if len(data['images']) == 0:
        logger.error("No data loaded. Training aborted.")
        return
    
    # Prepare data
    X = np.array(data['images'])
    y = np.array(data['labels'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Create model
    model = RobustAIImageDetector(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    # Create trainer
    trainer = RobustModelTrainer(model)
    
    # Train model
    best_val_acc = trainer.train(
        train_loader=None,  # Will be implemented with DataLoader
        val_loader=None,    # Will be implemented with DataLoader
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_train_loss': trainer.train_losses[-1],
        'final_val_loss': trainer.val_losses[-1],
        'final_train_acc': trainer.train_accuracies[-1],
        'final_val_acc': trainer.val_accuracies[-1],
        'total_samples': len(data['images']),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    with open('robust_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training results saved to robust_training_results.json")

if __name__ == "__main__":
    main()
