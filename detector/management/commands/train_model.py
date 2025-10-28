"""
Django management command for training AI image detection model
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import json
import logging
from pathlib import Path

class Command(BaseCommand):
    help = 'Train AI image detection model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--config',
            type=str,
            default='training_config.json',
            help='Path to training configuration file'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=None,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--max-samples',
            type=int,
            default=None,
            help='Maximum number of samples to use'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=None,
            help='Batch size for training'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=None,
            help='Learning rate for training'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='.',
            help='Output directory for model and plots'
        )

    def handle(self, *args, **options):
        """Handle the training command"""
        try:
            # Set up logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            
            # Load configuration
            config_path = options['config']
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file {config_path} not found, using defaults")
                config = {
                    "dataset": {"max_samples": 2000, "test_size": 0.2, "val_size": 0.1},
                    "model": {"backbone": "microsoft/resnet-50", "num_classes": 2, "dropout": 0.3},
                    "training": {"epochs": 5, "batch_size": 8, "learning_rate": 1e-4},
                    "paths": {"model_save_path": "trained_ai_detector.pth"}
                }
            
            # Override config with command line arguments
            if options['epochs'] is not None:
                config["training"]["epochs"] = options['epochs']
            if options['max_samples'] is not None:
                config["dataset"]["max_samples"] = options['max_samples']
            if options['batch_size'] is not None:
                config["training"]["batch_size"] = options['batch_size']
            if options['learning_rate'] is not None:
                config["training"]["learning_rate"] = options['learning_rate']
            
            # Reduce batch size and max samples for memory efficiency
            config["training"]["batch_size"] = min(config["training"]["batch_size"], 8)
            config["dataset"]["max_samples"] = min(config["dataset"]["max_samples"], 2000)
            
            logger.info(f"Training configuration: {config}")
            
            # Import training modules
            from detector.training import (
                DatasetProcessor, ModelTrainer, AIImageDetector, AIImageDataset
            )
            from torch.utils.data import DataLoader
            from transformers import AutoImageProcessor
            from sklearn.model_selection import train_test_split
            from tqdm import tqdm
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Initialize components
            processor = DatasetProcessor()
            
            # Load dataset
            logger.info("Loading dataset from Hugging Face...")
            if not processor.load_dataset():
                raise CommandError("Failed to load dataset")
            
            # Prepare data
            logger.info("Preparing training data...")
            images, labels = processor.prepare_data(max_samples=config["dataset"]["max_samples"])
            if not images:
                raise CommandError("Failed to prepare data")
            
            # Split data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(
                images, labels, 
                test_size=config["dataset"]["test_size"],
                val_size=config["dataset"]["val_size"]
            )
            
            # Initialize processor for images
            image_processor = AutoImageProcessor.from_pretrained(config["model"]["backbone"])
            
            # Create datasets
            train_dataset = AIImageDataset(X_train, y_train, image_processor)
            val_dataset = AIImageDataset(X_val, y_val, image_processor)
            test_dataset = AIImageDataset(X_test, y_test, image_processor)
            
            # Create data loaders with smaller batch sizes
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config["training"]["batch_size"], 
                shuffle=True,
                num_workers=0  # Disable multiprocessing to prevent memory issues
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config["training"]["batch_size"], 
                shuffle=False,
                num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config["training"]["batch_size"], 
                shuffle=False,
                num_workers=0
            )
            
            # Initialize model
            model = AIImageDetector(
                model_name=config["model"]["backbone"],
                num_classes=config["model"]["num_classes"],
                dropout=config["model"]["dropout"]
            )
            trainer = ModelTrainer(model)
            
            # Train model
            logger.info(f"Training model for {config['training']['epochs']} epochs...")
            trainer.train(
                train_loader, val_loader, 
                epochs=config["training"]["epochs"], 
                learning_rate=config["training"]["learning_rate"]
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            accuracy, report, predictions, labels = trainer.evaluate(test_loader)
            
            # Save model
            model_path = os.path.join(options['output_dir'], config["paths"]["model_save_path"])
            trainer.save_model(model_path)
            
            # Plot training history
            plots_dir = os.path.join(options['output_dir'], "training_plots")
            os.makedirs(plots_dir, exist_ok=True)
            trainer.plot_training_history(os.path.join(plots_dir, "training_history.png"))
            
            # Save training results
            results = {
                "test_accuracy": float(accuracy),
                "classification_report": report,
                "config": config,
                "training_losses": trainer.train_losses,
                "val_losses": trainer.val_losses,
                "train_accuracies": trainer.train_accuracies,
                "val_accuracies": trainer.val_accuracies
            }
            
            results_path = os.path.join(options['output_dir'], "training_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.stdout.write(
                self.style.SUCCESS(f"Training completed successfully!")
            )
            self.stdout.write(f"Test accuracy: {accuracy:.4f}")
            self.stdout.write(f"Model saved to: {model_path}")
            self.stdout.write(f"Results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise CommandError(f"Training failed: {str(e)}")
