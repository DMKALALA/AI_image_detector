"""
Fine-tune Hugging Face models on GenImage dataset
Supports all three models: ViT AI-detector, AI vs Human Detector, WildFakeDetector
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class GenImageDataset(Dataset):
    """Dataset for GenImage AI detection"""
    
    def __init__(self, data_file, processor):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class Command(BaseCommand):
    help = 'Fine-tune Hugging Face models on GenImage dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            choices=['vit', 'ai-human', 'wildfake', 'all'],
            default='all',
            help='Which model to fine-tune'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=5,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help='Training batch size'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=2e-5,
            help='Learning rate'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='hf_finetuned_models',
            help='Output directory for fine-tuned models'
        )

    def handle(self, *args, **options):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stdout.write(f'Using device: {device}')
        
        # Check if dataset is prepared
        splits_dir = Path('genimage_data/splits')
        if not splits_dir.exists():
            self.stdout.write(self.style.ERROR(
                'Dataset not prepared! Run: python manage.py prepare_genimage_dataset'
            ))
            return
        
        # Model configurations
        models_config = {
            'vit': {
                'name': 'dima806/deepfake_vs_real_image_detection',
                'processor_class': ViTImageProcessor,
                'model_class': ViTForImageClassification,
                'output_name': 'vit_ai_detector_finetuned'
            },
            'ai-human': {
                'name': 'umm-maybe/AI-image-detector',
                'processor_class': AutoFeatureExtractor,
                'model_class': AutoModelForImageClassification,
                'output_name': 'ai_human_detector_finetuned'
            },
            'wildfake': {
                'name': 'Aaditya2763/wild-fake-detector',
                'processor_class': AutoFeatureExtractor,
                'model_class': AutoModelForImageClassification,
                'output_name': 'wildfake_detector_finetuned'
            }
        }
        
        # Determine which models to train
        if options['model'] == 'all':
            models_to_train = list(models_config.keys())
        else:
            models_to_train = [options['model']]
        
        output_base = Path(options['output_dir'])
        output_base.mkdir(exist_ok=True)
        
        # Train each model
        for model_key in models_to_train:
            config = models_config[model_key]
            self.stdout.write(self.style.SUCCESS(
                f'\n{"="*60}\nFine-tuning: {config["name"]}\n{"="*60}'
            ))
            
            try:
                self._finetune_model(
                    model_name=config['name'],
                    processor_class=config['processor_class'],
                    model_class=config['model_class'],
                    output_dir=output_base / config['output_name'],
                    splits_dir=splits_dir,
                    epochs=options['epochs'],
                    batch_size=options['batch_size'],
                    learning_rate=options['learning_rate'],
                    device=device
                )
                self.stdout.write(self.style.SUCCESS(
                    f'✓ Successfully fine-tuned {model_key}'
                ))
            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'✗ Failed to fine-tune {model_key}: {e}'
                ))
                logger.error(f'Error fine-tuning {model_key}', exc_info=True)
    
    def _finetune_model(self, model_name, processor_class, model_class, output_dir,
                       splits_dir, epochs, batch_size, learning_rate, device):
        """Fine-tune a single model"""
        
        # Load processor and model
        self.stdout.write(f'Loading model: {model_name}')
        processor = processor_class.from_pretrained(model_name)
        model = model_class.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Create datasets
        self.stdout.write('Loading datasets...')
        train_dataset = GenImageDataset(
            splits_dir / 'train.json',
            processor
        )
        val_dataset = GenImageDataset(
            splits_dir / 'val.json',
            processor
        )
        
        self.stdout.write(f'Train samples: {len(train_dataset)}')
        self.stdout.write(f'Val samples: {len(val_dataset)}')
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            save_total_limit=2,
            report_to='none',
            fp16=torch.cuda.is_available(),
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        self.stdout.write('Starting training...')
        trainer.train()
        
        # Evaluate
        self.stdout.write('Evaluating on validation set...')
        metrics = trainer.evaluate()
        
        self.stdout.write(self.style.SUCCESS('Validation Results:'))
        self.stdout.write(f"  Accuracy:  {metrics['eval_accuracy']*100:.2f}%")
        self.stdout.write(f"  Precision: {metrics['eval_precision']*100:.2f}%")
        self.stdout.write(f"  Recall:    {metrics['eval_recall']*100:.2f}%")
        self.stdout.write(f"  F1 Score:  {metrics['eval_f1']*100:.2f}%")
        
        # Save final model
        self.stdout.write(f'Saving model to: {output_dir}')
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

