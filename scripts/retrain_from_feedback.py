#!/usr/bin/env python3
"""
Retrain AI Detection Models Using User Feedback Data
====================================================

This script fine-tunes the classification heads of the deep learning models
using the uploaded images and user feedback as ground truth labels.
"""

import os
import sys
import django
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import json
from datetime import datetime

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'image_detector_project.settings')
django.setup()

from detector.models import ImageUpload


class FeedbackDataset(Dataset):
    """Dataset from user feedback on uploaded images"""
    
    def __init__(self, image_uploads, transform=None):
        """
        Args:
            image_uploads: QuerySet of ImageUpload objects with user_feedback
            transform: Optional transform to apply to images
        """
        self.samples = []
        self.transform = transform
        
        for upload in image_uploads:
            if not upload.user_feedback or upload.user_feedback == 'unsure':
                continue
            
            # Determine ground truth from feedback
            if upload.user_feedback == 'correct':
                # Model was correct, use its prediction as label
                label = 1 if upload.is_ai_generated else 0
            elif upload.user_feedback == 'incorrect':
                # Model was wrong, flip the label
                label = 0 if upload.is_ai_generated else 1
            else:
                continue  # Skip unsure or other feedback
            
            # Get image path
            if upload.image and os.path.exists(upload.image.path):
                self.samples.append((upload.image.path, label))
        
        print(f"Loaded {len(self.samples)} samples from feedback")
        
        # Count distribution
        ai_count = sum(1 for _, label in self.samples if label == 1)
        real_count = len(self.samples) - ai_count
        print(f"  AI images: {ai_count} ({ai_count/len(self.samples)*100:.1f}%)")
        print(f"  Real images: {real_count} ({real_count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy sample
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (224, 224)))
            else:
                dummy_image = Image.new('RGB', (224, 224))
            return dummy_image, label


def create_model(model_name='efficientnet_b0', num_classes=2):
    """Create a model with pre-trained backbone and fresh classifier"""
    
    if model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1000)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    elif model_name == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1000)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'vit_large':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=1000)
        num_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True, num_classes=1000)
        num_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, model_name='model'):
    """Train a model with the given data"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Only train the classifier, freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            for param in model.head.fc.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.head.fc.parameters(), lr=lr)
        else:
            for param in model.head.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.head.parameters(), lr=lr)
    else:
        raise ValueError("Model has no classifier or head attribute")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nTraining {model_name}...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
        
        print()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc


def main():
    """Main training function"""
    
    print("="*60)
    print("AI IMAGE DETECTOR - RETRAINING FROM FEEDBACK")
    print("="*60)
    print()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 15
    learning_rate = 0.001
    val_split = 0.2
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation split: {val_split*100}%")
    print()
    
    # Load data from database
    print("Loading data from database...")
    uploads = ImageUpload.objects.exclude(
        user_feedback__isnull=True
    ).exclude(
        user_feedback='unsure'
    ).filter(
        image__isnull=False
    )
    
    print(f"Found {uploads.count()} uploads with usable feedback")
    
    if uploads.count() < 50:
        print("ERROR: Not enough training data (minimum 50 samples)")
        return
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = FeedbackDataset(list(uploads), transform=transform)
    
    # Split into train/val
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Models to train (start with lightweight one for testing)
    models_to_train = ['efficientnet_b0']
    
    # Check memory constraints
    memory_constrained = os.environ.get('MEMORY_CONSTRAINED', 'true').lower() == 'true'
    if not memory_constrained:
        models_to_train.extend(['efficientnet_b4', 'vit_large', 'convnext_base'])
    
    print(f"Training {len(models_to_train)} model(s): {', '.join(models_to_train)}")
    print()
    
    # Train each model
    results = {}
    trained_models = {}
    
    for model_name in models_to_train:
        try:
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            model = create_model(model_name)
            model, history, best_val_acc = train_model(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=learning_rate, model_name=model_name
            )
            
            results[model_name] = {
                'best_val_acc': best_val_acc,
                'history': history
            }
            
            trained_models[model_name] = model
            
            # Save model
            save_path = f'trained_models/{model_name}_finetuned.pth'
            os.makedirs('trained_models', exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'model_name': model_name,
                'training_date': datetime.now().isoformat(),
                'num_training_samples': len(train_dataset),
                'num_val_samples': len(val_dataset)
            }, save_path)
            
            print(f"\n✓ Model saved to: {save_path}")
            print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
            
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        print(f"  Final Train Accuracy: {result['history']['train_acc'][-1]:.2f}%")
        print(f"  Final Val Accuracy: {result['history']['val_acc'][-1]:.2f}%")
    
    # Save training results
    results_file = 'training_results.json'
    with open(results_file, 'w') as f:
        # Convert history to JSON-serializable format
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'best_val_acc': float(result['best_val_acc']),
                'final_train_acc': float(result['history']['train_acc'][-1]),
                'final_val_acc': float(result['history']['val_acc'][-1]),
                'training_date': datetime.now().isoformat()
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Training results saved to: {results_file}")
    print("\nNext steps:")
    print("1. Review training_results.json for accuracy metrics")
    print("2. Update improved_method_1_deeplearning.py to load trained models")
    print("3. Test the updated system with new images")
    print("4. Monitor accuracy improvements on new uploads")
    

if __name__ == '__main__':
    main()
