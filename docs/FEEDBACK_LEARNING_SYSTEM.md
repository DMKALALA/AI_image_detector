# Feedback Learning System

## Overview

The AI Image Detector includes a comprehensive feedback learning system that improves detection accuracy over time based on user feedback. The system has two main components:

1. **FeedbackLearningService** - Image-specific learning through hash-based re-upload detection
2. **AdaptiveLearningService** - System-wide method weight optimization based on aggregate feedback

## How It Works

### 1. FeedbackLearningService (Re-upload Detection)

Located in: `detector/feedback_learning.py`

This service tracks individual images by their content hash (MD5) and remembers user feedback for each image.

#### Key Features:

- **Image Hash Tracking**: Computes MD5 hash of image content to identify re-uploads
- **Feedback Memory**: Stores feedback in `feedback_memory.json` for persistence across server restarts
- **Prediction Override**: When a previously-seen image is uploaded again, the system:
  - Shows "🔄 Image seen X time(s) before" indicator
  - If user marked it as "incorrect", overrides the prediction with the learned correct answer
  - If user marked it as "correct", boosts confidence by 15%
  - If prediction was wrong, reduces confidence by 40%

#### Data Flow:

```
1. User uploads image → Detection runs → Result shown
2. User provides feedback (correct/incorrect/unsure)
3. Feedback recorded with image hash in feedback_memory.json
4. Same image uploaded again → Hash matches → Previous feedback found
5. Prediction adjusted based on learned correct answer
```

#### Feedback Memory Structure:

```json
{
  "3dca68f483733c67227540a874e91b6c": {
    "times_seen": 2,
    "prediction": false,
    "confidence": 0.72,
    "feedback": "incorrect",
    "correct_answer": true,
    "predictions": [
      {"prediction": false, "confidence": 0.72, "feedback": "incorrect"},
      {"prediction": true, "confidence": 0.86, "feedback": "correct"}
    ]
  }
}
```

### 2. AdaptiveLearningService (Method Weight Optimization)

Located in: `detector/adaptive_learning_service.py`

This service analyzes aggregate feedback to optimize the weights assigned to each detection method.

#### Key Features:

- **Automatic Weight Updates**: Analyzes feedback every 24 hours (configurable)
- **Minimum Sample Requirement**: Requires at least 20 feedback samples before updating (configurable)
- **Learning Rate**: Uses exponential moving average with 0.1 learning rate to smooth weight changes
- **Confidence Calibration**: Adjusts confidence calibration for overconfident/underconfident methods

#### Configuration (`adaptive_learning_config.json`):

```json
{
  "auto_update_enabled": true,
  "update_interval_hours": 24,
  "min_feedback_samples": 20,
  "learning_rate": 0.1,
  "last_update": "2025-12-12T00:37:50.803577+00:00"
}
```

#### Method Weights (`method_weights_config.json`):

```json
{
  "weights": {
    "method_1": 0.21,
    "method_2": 0.59,
    "method_3": 0.20
  },
  "calibration": {
    "method_1": 0.87,
    "method_2": 0.89,
    "method_3": 0.65
  }
}
```

## User Feedback Flow

### Submitting Feedback

1. User views detection result page
2. Clicks "Correct", "Incorrect", or "Unsure" button
3. System records feedback in:
   - Database (`ImageUpload.user_feedback` field)
   - Feedback memory (`feedback_memory.json`) with image hash

### Feedback Processing

When feedback is submitted (`detector/views.py:submit_feedback`):

```python
# 1. Save to database
image_upload.save_feedback(feedback, notes)

# 2. Record in feedback learning service (hash-based)
feedback_service.record_feedback(
    image_hash=image_hash,
    prediction=image_upload.is_ai_generated,
    confidence=image_upload.confidence_score,
    user_feedback=feedback,
    correct_answer=correct_answer
)

# 3. Trigger adaptive learning check
adaptive_learning_service.trigger_learning_on_feedback(image_upload)
```

## Technical Details

### Type Conversion Fix

The system converts numpy types to native Python types before JSON serialization to prevent `"Object of type bool_ is not JSON serializable"` errors:

```python
# In record_feedback()
prediction = bool(prediction) if prediction is not None else None
confidence = float(confidence) if confidence is not None else 0.0
correct_answer = bool(correct_answer) if correct_answer is not None else None
```

### Confidence Adjustments

| Feedback Type | Adjustment |
|--------------|------------|
| `correct` | Confidence × 1.15 (max 0.98) |
| `incorrect` (with override) | Confidence × 1.20 (max 0.95) |
| `incorrect` (no override) | Confidence × 0.60 |

### Hash Computation

Uses MD5 hash of raw image bytes for content-based identification:

```python
def compute_image_hash(self, image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
```

## Files

| File | Purpose |
|------|---------|
| `detector/feedback_learning.py` | FeedbackLearningService and GeneralizedFeedbackLearning classes |
| `detector/adaptive_learning_service.py` | AdaptiveLearningService class |
| `detector/views.py` | `submit_feedback` endpoint |
| `feedback_memory.json` | Persistent storage for image-specific feedback |
| `adaptive_learning_config.json` | Configuration for adaptive learning |
| `method_weights_config.json` | Current method weights and calibration |

## Monitoring

### Feedback Statistics

View feedback statistics at `/feedback-stats/` endpoint showing:
- Total images analyzed
- Feedback rate
- Accuracy based on feedback
- Correct/Incorrect/Unsure counts

### Analytics Dashboard

View comprehensive analytics at `/analytics/` including:
- Method performance breakdown
- Confidence distribution
- Recent activity
- Model availability status

## Troubleshooting

### Feedback Not Being Applied

1. Check if `feedback_memory.json` exists and contains entries
2. Verify image hash matches (same file content)
3. Check server logs for "Feedback learning adjustment failed" errors

### Weights Not Updating

1. Check `adaptive_learning_config.json` for `last_update` timestamp
2. Verify at least 20 feedback samples exist
3. Check if `auto_update_enabled` is `true`
4. Wait 24 hours since last update or manually trigger via management command

### JSON Serialization Errors

If you see `"Object of type bool_ is not JSON serializable"`:
- Ensure numpy types are converted to native Python types
- Check that all values stored in feedback_memory are serializable
