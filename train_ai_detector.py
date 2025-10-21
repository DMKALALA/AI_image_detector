#!/usr/bin/env python3
"""
AI Image Detection Model Training Script

This script trains a custom model using the twitter_AII dataset from Hugging Face
to detect AI-generated images vs real images.

Usage:
    python train_ai_detector.py --epochs 10 --max-samples 10000
"""

import argparse
import json
import os
import logging
from pathlib import Path

def setup_logging(log_dir="training_logs"):
    """Set up logging for training"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train AI Image Detection Model')
    parser.add_argument('--config', type=str, default='training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for training')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for model and plots')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        config = {
            "dataset": {"max_samples": 5000, "test_size": 0.2, "val_size": 0.1},
            "model": {"backbone": "microsoft/resnet-50", "num_classes": 2, "dropout": 0.3},
            "training": {"epochs": 5, "batch_size": 16, "learning_rate": 1e-4},
            "paths": {"model_save_path": "trained_ai_detector.pth"}
        }
    
    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_samples is not None:
        config["dataset"]["max_samples"] = args.max_samples
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    
    logger.info(f"Training configuration: {config}")
    
    try:
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
            logger.error("Failed to load dataset")
            return 1
        
        # Prepare data
        logger.info("Preparing training data...")
        images, labels = processor.prepare_data(max_samples=config["dataset"]["max_samples"])
        if not images:
            logger.error("Failed to prepare data")
            return 1
        
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=False
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
        model_path = os.path.join(args.output_dir, config["paths"]["model_save_path"])
        trainer.save_model(model_path)
        
        # Plot training history
        plots_dir = os.path.join(args.output_dir, "training_plots")
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
        
        results_path = os.path.join(args.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
