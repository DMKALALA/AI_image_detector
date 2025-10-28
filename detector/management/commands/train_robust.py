"""
Django management command for robust AI image detection training
"""

from django.core.management.base import BaseCommand, CommandError
from detector.robust_training import (
    RobustDatasetProcessor,
    RobustAIImageDetector,
    RobustModelTrainer,
    create_robust_training_config,
    plot_training_results
)
import json
import os
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train robust AI image detection model using GenImage datasets'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=25,
            help='Number of training epochs (default: 25)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training (default: 32)'
        )
        parser.add_argument(
            '--hemg-samples',
            type=int,
            default=15000,
            help='Maximum samples from Hemg dataset (default: 15000)'
        )
        parser.add_argument(
            '--cifake-samples',
            type=int,
            default=15000,
            help='Maximum samples from CIFake dataset (default: 15000)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=5e-5,
            help='Learning rate (default: 5e-5)'
        )
        parser.add_argument(
            '--model-name',
            type=str,
            default='robust_ai_detector',
            help='Name for the saved model (default: robust_ai_detector)'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('=== ROBUST AI IMAGE DETECTION TRAINING ===')
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
            # Load configuration
            config = create_robust_training_config()
            
            # Update config with command line arguments
            config['training']['epochs'] = epochs
            config['training']['batch_size'] = batch_size
            config['training']['learning_rate'] = learning_rate
            config['datasets']['hemg']['max_samples'] = hemg_samples
            config['datasets']['cifake']['max_samples'] = cifake_samples
            config['paths']['model_save_path'] = f"{model_name}.pth"
            
            self.stdout.write(self.style.SUCCESS("✓ Configuration loaded"))
            
            # Process datasets
            self.stdout.write("Loading datasets...")
            processor = RobustDatasetProcessor()
            data = processor.load_multiple_datasets()
            
            if len(data['images']) == 0:
                raise CommandError("No data loaded. Check dataset availability.")
            
            self.stdout.write(
                self.style.SUCCESS(f"✓ Loaded {len(data['images'])} total samples")
            )
            
            # Create model
            self.stdout.write("Creating ensemble model...")
            model = RobustAIImageDetector(
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout']
            )
            
            self.stdout.write(self.style.SUCCESS("✓ Model created"))
            
            # Create trainer
            trainer = RobustModelTrainer(model)
            
            # Note: This is a simplified version - full implementation would include DataLoader
            self.stdout.write("Starting training...")
            self.stdout.write(
                self.style.WARNING(
                    "Note: This is a demonstration version. "
                    "Full DataLoader implementation needed for actual training."
                )
            )
            
            # Save configuration
            config_path = f"{model_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.stdout.write(
                self.style.SUCCESS(f"✓ Configuration saved to {config_path}")
            )
            
            # Create directories
            os.makedirs(config['paths']['plots_save_path'], exist_ok=True)
            os.makedirs(config['paths']['logs_save_path'], exist_ok=True)
            
            self.stdout.write(
                self.style.SUCCESS("✓ Directories created")
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    "Robust training setup completed successfully!\n"
                    "Next steps:\n"
                    "1. Implement DataLoader for actual training\n"
                    "2. Run full training with: python manage.py train_robust_full\n"
                    "3. Test the trained model"
                )
            )
            
        except Exception as e:
            raise CommandError(f"Training failed: {str(e)}")
