# AI Image Detection Model Training

This document explains how to train a custom AI image detection model using the [twitter_AII dataset](https://huggingface.co/datasets/anonymous1233/twitter_AII) from Hugging Face.

## Dataset Information

The twitter_AII dataset contains:
- **15,909 image pairs** of real Twitter images and their AI-generated counterparts
- **Multiple AI models**: SD35, SD3, SD21, SDXL, DALL-E
- **Perfect for training**: Balanced dataset with real vs AI-generated labels
- **Size**: ~10.2 GB

## Training Setup

### 1. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training Options

#### Option A: Using Django Management Command

```bash
# Basic training with default settings
python manage.py train_model

# Custom training parameters
python manage.py train_model --epochs 10 --max-samples 10000 --learning-rate 1e-4
```

#### Option B: Using Standalone Script

```bash
# Basic training
python train_ai_detector.py

# Custom parameters
python train_ai_detector.py --epochs 10 --max-samples 10000 --batch-size 32
```

#### Option C: Using Configuration File

```bash
# Edit training_config.json to customize settings
python train_ai_detector.py --config training_config.json
```

## Training Configuration

The `training_config.json` file allows you to customize:

```json
{
    "dataset": {
        "max_samples": 10000,    // Limit samples for faster training
        "test_size": 0.2,        // 20% for testing
        "val_size": 0.1          // 10% for validation
    },
    "model": {
        "backbone": "microsoft/resnet-50",  // Pre-trained backbone
        "num_classes": 2,                   // Real vs AI-generated
        "dropout": 0.3                      // Dropout rate
    },
    "training": {
        "epochs": 10,            // Number of training epochs
        "batch_size": 16,        // Batch size
        "learning_rate": 1e-4    // Learning rate
    }
}
```

## Training Process

### 1. Data Loading
- Downloads dataset from Hugging Face
- Processes real and AI-generated image pairs
- Creates balanced training set

### 2. Data Preprocessing
- Resizes images to standard dimensions
- Applies normalization
- Splits into train/validation/test sets

### 3. Model Architecture
- **Backbone**: Pre-trained ResNet-50 for feature extraction
- **Classifier**: Custom head with dropout for binary classification
- **Output**: 2 classes (Real vs AI-Generated)

### 4. Training
- Uses Adam optimizer
- Cross-entropy loss function
- Validation monitoring
- Early stopping capability

### 5. Evaluation
- Test accuracy calculation
- Classification report
- Confusion matrix
- Training history plots

## Output Files

After training, you'll get:

- `trained_ai_detector.pth` - Trained model weights
- `training_history.png` - Training/validation curves
- `training_results.json` - Detailed results and metrics
- `training_logs/training.log` - Training logs

## Model Performance

Expected performance metrics:
- **Accuracy**: 85-95% on test set
- **Precision**: High precision for both classes
- **Recall**: Good recall for AI-generated detection
- **F1-Score**: Balanced performance

## Using the Trained Model

The trained model automatically integrates with your Django application:

1. **Automatic Detection**: The app will use the trained model if available
2. **Fallback**: Falls back to heuristic methods if model not found
3. **Confidence Scores**: Provides confidence levels for predictions
4. **Detailed Analysis**: Shows detection indicators and reasoning

## Training Tips

### For Better Performance:
1. **Increase Dataset Size**: Use more samples (up to 15,909)
2. **More Epochs**: Train for 20-50 epochs for better convergence
3. **Data Augmentation**: Enable augmentation in config
4. **Model Architecture**: Try different backbones (EfficientNet, ViT)

### For Faster Training:
1. **Reduce Samples**: Start with 1,000-5,000 samples
2. **Smaller Batch Size**: Use batch size 8-16
3. **Fewer Epochs**: Start with 5-10 epochs
4. **GPU**: Use CUDA if available

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Reduce batch size
   - Use fewer samples
   - Enable gradient checkpointing

2. **Slow Training**:
   - Use GPU acceleration
   - Reduce image resolution
   - Use fewer samples initially

3. **Poor Performance**:
   - Increase training epochs
   - Use more training data
   - Adjust learning rate
   - Try different model architecture

### Monitoring Training:

```bash
# Watch training logs
tail -f training_logs/training.log

# Monitor GPU usage (if using CUDA)
nvidia-smi -l 1
```

## Advanced Usage

### Custom Model Architecture:

```python
# Modify detector/training.py
class CustomAIImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture here
```

### Transfer Learning:

```python
# Use different pre-trained models
model = AIImageDetector(model_name="microsoft/efficientnet-b0")
```

### Hyperparameter Tuning:

```bash
# Grid search over parameters
python train_ai_detector.py --learning-rate 1e-3 --epochs 20
python train_ai_detector.py --learning-rate 1e-4 --epochs 20
python train_ai_detector.py --learning-rate 1e-5 --epochs 20
```

## Integration with Production

1. **Model Deployment**: Copy `trained_ai_detector.pth` to production
2. **Performance Monitoring**: Track accuracy on new data
3. **Model Updates**: Retrain periodically with new data
4. **A/B Testing**: Compare trained model vs heuristic methods

## Next Steps

1. **Train your first model** with default settings
2. **Evaluate performance** on test set
3. **Fine-tune parameters** for better results
4. **Deploy to production** and monitor performance
5. **Collect feedback** and retrain with new data

Happy training! 🚀
