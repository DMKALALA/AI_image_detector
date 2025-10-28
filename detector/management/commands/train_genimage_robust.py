"""
Django management command for GenImage-specific robust training
"""

from django.core.management.base import BaseCommand, CommandError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import logging

logger = logging.getLogger(__name__)

class GenImageDataset(Dataset):
    """Custom dataset for GenImage data"""
    
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

class SimpleGenImageDetector(nn.Module):
    """Simplified GenImage detector using ResNet50"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(SimpleGenImageDetector, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class Command(BaseCommand):
    help = 'Train GenImage AI detection model with simplified architecture'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='Number of training epochs (default: 10)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=16,
            help='Batch size for training (default: 16)'
        )
        parser.add_argument(
            '--hemg-samples',
            type=int,
            default=5000,
            help='Maximum samples from Hemg dataset (default: 5000)'
        )
        parser.add_argument(
            '--cifake-samples',
            type=int,
            default=5000,
            help='Maximum samples from CIFake dataset (default: 5000)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-4,
            help='Learning rate (default: 1e-4)'
        )
        parser.add_argument(
            '--model-name',
            type=str,
            default='genimage_detector',
            help='Name for the saved model (default: genimage_detector)'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('=== GENIMAGE AI DETECTION TRAINING ===')
        )
        
        # Get parameters
        epochs = options['epochs']
        batch_size = options['batch_size']
        hemg_samples = options['hemg_samples']
        cifake_samples = options['cifake_samples']
        learning_rate = options['learning_rate']
        model_name = options['model_name']
        
        self.stdout.write(f"Configuration:")
        self.stdout.write(f"  Epochs: {epochs}")
        self.stdout.write(f"  Batch Size: {batch_size}")
        self.stdout.write(f"  Hemg Samples: {hemg_samples}")
        self.stdout.write(f"  CIFake Samples: {cifake_samples}")
        self.stdout.write(f"  Learning Rate: {learning_rate}")
        self.stdout.write(f"  Model Name: {model_name}")
        
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.stdout.write(f"Using device: {device}")
            
            # Load datasets
            self.stdout.write("Loading GenImage datasets...")
            
            images = []
            labels = []
            
            # Try to load Hemg dataset
            try:
                self.stdout.write("Loading Hemg/AI-Generated-vs-Real-Images-Datasets...")
                hemg_dataset = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets')
                
                train_data = hemg_dataset['train']
                num_samples = min(hemg_samples, len(train_data))
                train_data = train_data.select(range(num_samples))
                
                for i, sample in enumerate(tqdm(train_data, desc="Processing Hemg")):
                    try:
                        if 'image' in sample:
                            image = sample['image']
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            image = image.resize((224, 224))
                            images.append(np.array(image))
                            
                            # Label: 0 for real, 1 for AI
                            label = sample.get('label', 0)
                            labels.append(label)
                    except Exception as e:
                        logger.warning(f"Failed to process Hemg sample {i}: {e}")
                        continue
                        
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Loaded {len(images)} samples from Hemg dataset")
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Failed to load Hemg dataset: {e}")
                )
            
            # Try to load CIFake dataset
            try:
                self.stdout.write("Loading CIFake dataset...")
                cifake_dataset = load_dataset('cifake')
                
                train_data = cifake_dataset['train']
                num_samples = min(cifake_samples, len(train_data))
                train_data = train_data.select(range(num_samples))
                
                start_idx = len(images)
                
                for i, sample in enumerate(tqdm(train_data, desc="Processing CIFake")):
                    try:
                        if 'image' in sample:
                            image = sample['image']
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            image = image.resize((224, 224))
                            images.append(np.array(image))
                            
                            # Label: 0 for real, 1 for AI
                            label = sample.get('label', 0)
                            labels.append(label)
                    except Exception as e:
                        logger.warning(f"Failed to process CIFake sample {i}: {e}")
                        continue
                        
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Loaded {len(images) - start_idx} samples from CIFake dataset")
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Failed to load CIFake dataset: {e}")
                )
            
            if len(images) == 0:
                raise CommandError("No data loaded. Check dataset availability.")
            
            self.stdout.write(
                self.style.SUCCESS(f"✓ Total samples loaded: {len(images)}")
            )
            
            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            
            self.stdout.write(f"Data split:")
            self.stdout.write(f"  Training: {len(X_train)} samples")
            self.stdout.write(f"  Validation: {len(X_val)} samples")
            self.stdout.write(f"  Test: {len(X_test)} samples")
            
            # Define transforms
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create datasets
            train_dataset = GenImageDataset(X_train, y_train, train_transform)
            val_dataset = GenImageDataset(X_val, y_val, val_transform)
            test_dataset = GenImageDataset(X_test, y_test, val_transform)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Create model
            self.stdout.write("Creating model...")
            model = SimpleGenImageDetector(num_classes=2, dropout=0.3)
            model = model.to(device)
            
            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            # Training loop
            self.stdout.write("Starting training...")
            
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            best_val_acc = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
                    train_total += target.size(0)
                
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        pred = output.argmax(dim=1, keepdim=True)
                        val_correct += pred.eq(target.view_as(pred)).sum().item()
                        val_total += target.size(0)
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                self.stdout.write(f"Epoch {epoch+1}/{epochs}:")
                self.stdout.write(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                self.stdout.write(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f"{model_name}_best.pth")
                    self.stdout.write(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
                
                scheduler.step()
            
            # Final test evaluation
            self.stdout.write("Evaluating on test set...")
            model.load_state_dict(torch.load(f"{model_name}_best.pth"))
            model.eval()
            
            test_correct = 0
            test_total = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
                    
                    all_preds.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy())
            
            test_acc = 100. * test_correct / test_total
            
            self.stdout.write(
                self.style.SUCCESS(f"✓ Final Test Accuracy: {test_acc:.2f}%")
            )
            self.stdout.write(
                self.style.SUCCESS(f"✓ Best Validation Accuracy: {best_val_acc:.2f}%")
            )
            
            # Save results
            results = {
                'model_name': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'total_samples': len(images),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'best_val_accuracy': best_val_acc,
                'final_test_accuracy': test_acc,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_train_acc': train_accuracies[-1],
                'final_val_acc': val_accuracies[-1]
            }
            
            with open(f"{model_name}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            self.stdout.write(
                self.style.SUCCESS(f"✓ Results saved to {model_name}_results.json")
            )
            
            # Plot training results
            try:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracies, label='Train Accuracy')
                plt.plot(val_accuracies, label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{model_name}_training_plots.png")
                plt.close()
                
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Training plots saved to {model_name}_training_plots.png")
                )
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Failed to save plots: {e}")
                )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n🎉 GenImage training completed successfully!\n"
                    f"📊 Best Validation Accuracy: {best_val_acc:.2f}%\n"
                    f"📊 Final Test Accuracy: {test_acc:.2f}%\n"
                    f"💾 Model saved as: {model_name}_best.pth\n"
                    f"📈 Results saved as: {model_name}_results.json"
                )
            )
            
        except Exception as e:
            raise CommandError(f"Training failed: {str(e)}")
