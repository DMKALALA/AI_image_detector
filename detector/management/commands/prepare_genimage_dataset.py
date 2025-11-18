"""
Prepare GenImage dataset for fine-tuning Hugging Face models
Creates train/val/test splits and saves metadata
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import json
import random
from PIL import Image
import shutil


class Command(BaseCommand):
    help = 'Prepare GenImage dataset for fine-tuning'

    def add_arguments(self, parser):
        parser.add_argument(
            '--val-split',
            type=float,
            default=0.15,
            help='Validation split ratio (default: 0.15)'
        )
        parser.add_argument(
            '--test-split',
            type=float,
            default=0.15,
            help='Test split ratio (default: 0.15)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed for reproducibility'
        )

    def handle(self, *args, **options):
        random.seed(options['seed'])
        
        base_dir = Path('genimage_data')
        ai_dir = base_dir / 'ai_images'
        real_dir = base_dir / 'real_images'
        
        if not ai_dir.exists() or not real_dir.exists():
            self.stdout.write(self.style.ERROR('GenImage data not found!'))
            return
        
        # Collect all images
        ai_images = list(ai_dir.glob('*.jpg')) + list(ai_dir.glob('*.png'))
        real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
        
        self.stdout.write(f'Found {len(ai_images)} AI images')
        self.stdout.write(f'Found {len(real_images)} real images')
        
        # Create dataset entries
        dataset = []
        for img_path in ai_images:
            dataset.append({
                'path': str(img_path),
                'label': 1,  # AI-generated
                'label_name': 'ai'
            })
        
        for img_path in real_images:
            dataset.append({
                'path': str(img_path),
                'label': 0,  # Real
                'label_name': 'real'
            })
        
        # Shuffle
        random.shuffle(dataset)
        
        # Split
        val_split = options['val_split']
        test_split = options['test_split']
        
        n = len(dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_test - n_val
        
        train_data = dataset[:n_train]
        val_data = dataset[n_train:n_train+n_val]
        test_data = dataset[n_train+n_val:]
        
        # Create splits directory
        splits_dir = base_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        # Save splits
        with open(splits_dir / 'train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(splits_dir / 'val.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(splits_dir / 'test.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'total': n,
            'train': n_train,
            'val': n_val,
            'test': n_test,
            'ai_images': len(ai_images),
            'real_images': len(real_images),
            'val_split': val_split,
            'test_split': test_split,
            'seed': options['seed']
        }
        
        with open(splits_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS('✓ Dataset prepared successfully!'))
        self.stdout.write(f'  Train: {n_train} samples')
        self.stdout.write(f'  Val:   {n_val} samples')
        self.stdout.write(f'  Test:  {n_test} samples')
        self.stdout.write(f'  Splits saved to: {splits_dir}')

