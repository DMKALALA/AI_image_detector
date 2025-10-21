from django.core.management.base import BaseCommand
from detector.training import main as train_model
import logging

class Command(BaseCommand):
    help = 'Train the AI image detection model using the twitter_AII dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=5,
            help='Number of training epochs (default: 5)'
        )
        parser.add_argument(
            '--max-samples',
            type=int,
            default=5000,
            help='Maximum number of samples to use for training (default: 5000)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-4,
            help='Learning rate for training (default: 1e-4)'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting AI image detection model training...')
        )
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Import and run training
            from detector.training import DatasetProcessor, ModelTrainer, AIImageDetector
            from detector.training import AIImageDataset
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
            self.stdout.write('Loading dataset from Hugging Face...')
            if not processor.load_dataset():
                self.stdout.write(
                    self.style.ERROR('Failed to load dataset')
                )
                return
            
            # Prepare data
            self.stdout.write('Preparing training data...')
            images, labels = processor.prepare_data(max_samples=options['max_samples'])
            if not images:
                self.stdout.write(
                    self.style.ERROR('Failed to prepare data')
                )
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
            self.stdout.write(f'Training model for {options["epochs"]} epochs...')
            trainer.train(train_loader, val_loader, 
                        epochs=options['epochs'], 
                        learning_rate=options['learning_rate'])
            
            # Evaluate model
            self.stdout.write('Evaluating model...')
            accuracy, report, predictions, labels = trainer.evaluate(test_loader)
            
            # Save model
            trainer.save_model('trained_ai_detector.pth')
            
            # Plot training history
            trainer.plot_training_history('training_history.png')
            
            self.stdout.write(
                self.style.SUCCESS(f'Training completed successfully! Test accuracy: {accuracy:.4f}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Training failed: {str(e)}')
            )
            raise
