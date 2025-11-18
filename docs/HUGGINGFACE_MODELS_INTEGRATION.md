# Hugging Face Models Integration (Method 4)

## Overview

Added three specialized Hugging Face models for AI-generated image detection as **Method 4** in the detection pipeline. These models complement the existing three methods (Deep Learning, Statistical Analysis, and Forensics).

## Models Integrated

### 1. ViT AI-Image Detector
- **Model:** `dima806/deepfake_vs_real_image_detection`
- **Architecture:** Vision Transformer (ViT)
- **Purpose:** Specialized for AI-generated vs real image classification
- **Strength:** Strong global pattern recognition

### 2. AI vs Human Image Detector
- **Model:** `umm-maybe/AI-image-detector`
- **Purpose:** Binary classification (AI vs Human-created)
- **Strength:** Trained specifically for AI detection task

### 3. WildFakeDetector
- **Model:** `Aaditya2763/wild-fake-detector`
- **Purpose:** Diverse AI-generated image detection
- **Strength:** Trained on varied generative models (broader perspective)

## Implementation Details

### File Structure
```
detector/
├── huggingface_models.py       # NEW: HF model ensemble
└── three_method_detection_service.py  # Updated: Method 4 integration
```

### Key Components

#### `HuggingFaceEnsemble` Class
- Loads all three models on initialization
- Graceful degradation if models fail to load
- Weighted ensemble voting (equal weights by default)
- Returns combined prediction + individual model results

#### Detection Flow
1. Image preprocessed for each model
2. Each model returns AI/Real probabilities
3. Weighted average computed across available models
4. Final prediction with confidence score

### Weighted Voting Integration

#### Default Weights (4-method system)
```python
{
    'method_1': 0.35,  # Deep Learning (reduced from 0.50)
    'method_2': 0.30,  # Statistical (reduced from 0.40)
    'method_3': 0.10,  # Forensics
    'method_4': 0.25   # NEW: Hugging Face models
}
```

#### Confidence Calibration
```python
{
    'method_4': 0.95   # High confidence in pre-trained specialists
}
```

## Usage

### Automatic (Integrated)
When you upload an image, Method 4 automatically runs alongside Methods 1-3:

```python
# In views.py
result = get_detection_service().detect_ai_image(image_upload.image.path)
```

### Result Structure
```json
{
  "method_comparison": {
    "method_4": {
      "name": "Hugging Face Specialized Models",
      "description": "ViT AI-detector, AI vs Human Detector, WildFakeDetector",
      "is_ai_generated": true,
      "confidence": 0.87,
      "indicators": [
        "Hugging Face Ensemble: AI-generated",
        "Ensemble AI probability: 87.3%",
        "Models used: 3/3",
        "ViT AI-detector: AI (AI: 85.2%, confidence: 85.2%)",
        "AI vs Human Detector: AI (AI: 91.4%, confidence: 91.4%)",
        "WildFakeDetector: AI (AI: 85.3%, confidence: 85.3%)"
      ],
      "available": true,
      "model_predictions": {
        "vit_ai_detector": {...},
        "ai_human_detector": {...},
        "wildfake_detector": {...}
      }
    }
  }
}
```

## Benefits

1. **Specialized Training**: Models trained specifically for AI detection (vs general-purpose)
2. **Diverse Perspectives**: Three different architectures/training sets
3. **Ensemble Robustness**: Averaged predictions reduce individual model biases
4. **Pre-trained**: No custom training required, ready to use
5. **Fallback Safety**: System works even if HF models fail to load

## Performance Considerations

### Memory Usage
- **Free Tier (512MB)**: May struggle to load all 3 HF models
- **Recommended**: ≥2GB RAM for smooth operation
- **Fallback**: Models load individually; partial ensemble still useful

### Inference Speed
- ~2-5 seconds per image (3 models)
- Can be optimized with:
  - Model quantization
  - Batch processing
  - GPU acceleration

## Testing

### Quick Test
```bash
# Start server
python manage.py runserver

# Upload test image via UI at http://127.0.0.1:8000/
# Check result page for "Method 4: Hugging Face Specialized Models"
```

### API Test
```bash
curl -X POST http://127.0.0.1:8000/detector/api/detect/realtime/ \
  -F "image=@test_image.jpg" | jq '.analysis_details.method_comparison.method_4'
```

## Troubleshooting

### Model Loading Fails
- **Symptom**: "Hugging Face models not available" in indicators
- **Solution**: Check internet connection (models download on first use)
- **Alternative**: Pre-download models with:
  ```python
  from transformers import AutoModelForImageClassification
  AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
  ```

### Memory Issues
- Set `MEMORY_CONSTRAINED=true` to prevent loading
- Reduce number of models in `huggingface_models.py` (comment out lines)

### Low Confidence
- HF models may be conservative on edge cases
- Check individual model predictions in `model_predictions`
- Adjust `method_4` weight in `method_weights_config.json`

## Next Steps

1. **Evaluate Performance**: Run on test dataset, measure accuracy
2. **Fine-tune Weights**: Adjust Method 4 weight based on real data
3. **Add More Models**: Extend with additional HF models if needed
4. **Optimize Speed**: Implement model quantization or caching
5. **Update UI**: Display Method 4 results prominently in result page

## Commit

Committed to branch: `feature/provenance-synthid-detection`
Commit hash: `e933a9e`

To merge to main:
```bash
git checkout main
git merge feature/provenance-synthid-detection
git push origin main
```

