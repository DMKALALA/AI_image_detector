# Adaptive Learning System Documentation

## Overview

The Adaptive Learning System automatically improves the AI detector's accuracy by learning from user feedback. It continuously adjusts method weights and confidence calibration based on real-world performance data.

## 🎯 How It Works

### 1. **Feedback Collection**
- Every time a user provides feedback (correct/incorrect), it's stored in the database
- The system tracks which method made which prediction for each image

### 2. **Performance Analysis**
- Analyzes recent feedback (default: last 100 samples, minimum 20 required)
- Calculates accuracy for each detection method:
  - Method 1: Improved Deep Learning
  - Method 2: Statistical Pattern Analysis
  - Method 3: Advanced Forensics
- Tracks average confidence levels for each method

### 3. **Automatic Weight Adjustment**
- **Method Weights**: Automatically redistributes weights based on accuracy
  - High-accuracy methods get higher weights
  - Low-accuracy methods get lower weights
  - Uses exponential function to emphasize differences
  
- **Confidence Calibration**: Adjusts confidence multipliers
  - If a method is overconfident (high confidence, low accuracy), reduces calibration
  - If a method is underconfident (low confidence, high accuracy), increases calibration

### 4. **Learning Rate**
- Uses a learning rate (default: 0.1 = 10%) to blend old and new weights
- Prevents sudden, drastic changes
- Smoothly adapts over time

### 5. **Update Schedule**
- **Automatic**: Updates automatically when:
  - At least 20 feedback samples are available
  - 24 hours have passed since last update
  - New feedback is submitted (checks trigger update)
  
- **Manual**: Can be triggered manually via management command

---

## 🔧 Configuration

Configuration is stored in `adaptive_learning_config.json`:

```json
{
  "auto_update_enabled": true,
  "update_interval_hours": 24,
  "min_feedback_samples": 20,
  "learning_rate": 0.1,
  "last_update": "2025-10-28T15:00:00"
}
```

### Parameters:
- **`auto_update_enabled`**: Enable/disable automatic updates
- **`update_interval_hours`**: Minimum hours between automatic updates
- **`min_feedback_samples`**: Minimum feedback samples required before updating
- **`learning_rate`**: How aggressively to adjust weights (0.0-1.0)
  - `0.0` = No learning (keep old weights)
  - `1.0` = Full learning (replace with new weights)
  - `0.1` = Conservative (blend 10% new, 90% old)

---

## 📊 Current Performance (Latest Analysis)

Based on 50 recent feedback samples:

| Method | Accuracy | Weight | Status |
|--------|----------|--------|--------|
| **Method 1** (Deep Learning) | 30.0% | 12% → **5%** (recommended) | ❌ Needs improvement |
| **Method 2** (Statistical) | **92.0%** | 80% → **85%** (recommended) | ✅ Excellent |
| **Method 3** (Forensics) | 56.0% | 8% → **10%** (recommended) | ⚠️ Improving |

**Overall System Accuracy**: **92.0%** ✅

---

## 🚀 Usage

### Automatic Learning

**The system runs automatically!** No action needed:
- Every time a user submits feedback, the system checks if an update is needed
- Updates happen automatically every 24 hours (if enough feedback exists)

### Manual Learning Update

To manually trigger a learning update:

```bash
# Standard update (respects time interval)
python manage.py adaptive_learn

# Force update (ignores time interval)
python manage.py adaptive_learn --force

# Specify number of samples to analyze
python manage.py adaptive_learn --limit 200
```

### View Current Configuration

The system logs updates to the Django log. Check logs for:
- Current method performance
- Weight changes
- Update timestamps

---

## 📈 Learning Process Example

### Before Update:
```
Method 1: 30% accuracy → Weight: 12%
Method 2: 92% accuracy → Weight: 80%
Method 3: 56% accuracy → Weight: 8%
```

### Analysis:
- Method 2 is performing excellently (92%)
- Method 1 is underperforming (30%)
- Method 3 is improving (56%)

### After Update (with 0.1 learning rate):
```
Method 1: 30% accuracy → Weight: 11% (slight decrease)
Method 2: 92% accuracy → Weight: 81% (slight increase)
Method 3: 56% accuracy → Weight: 8% (stable)
```

### Over Multiple Updates:
The system gradually shifts weights toward better-performing methods, improving overall accuracy.

---

## 🔍 How Feedback Is Used

### Example Scenario:

1. **User uploads image** → Methods analyze it
2. **Method 1 predicts**: AI-Generated (50% confidence)
3. **Method 2 predicts**: Real (85% confidence)
4. **Method 3 predicts**: Real (66% confidence)
5. **Final decision**: Real (weighted majority)
6. **User feedback**: ❌ Incorrect (it was actually AI-Generated)

### Learning Process:
1. System records that **Method 1 was correct** (predicted AI)
2. System records that **Method 2 was wrong** (predicted Real)
3. System records that **Method 3 was wrong** (predicted Real)
4. Updates accuracy counts for each method
5. After enough feedback, recalculates weights:
   - Method 1 accuracy increases → weight increases slightly
   - Method 2/3 accuracy decreases → weights decrease slightly

---

## ⚙️ Technical Details

### Weight Calculation Formula

Uses a softmax-like function to convert accuracies to weights:

```python
# Normalize accuracies
normalized = {method: accuracy / total_accuracy for method, accuracy in accuracies}

# Apply exponential to emphasize differences
exp_accuracies = {method: exp(acc * 5) for method, acc in normalized}

# Calculate weights
weights = {method: exp_acc / sum(exp_accuracies) for method, exp_acc in exp_accuracies}
```

### Exponential Moving Average

When updating weights, uses exponential moving average to smooth changes:

```python
new_weight = (1 - learning_rate) * old_weight + learning_rate * optimal_weight
```

This prevents sudden changes and ensures stability.

### Confidence Calibration

Adjusts confidence multipliers based on accuracy/confidence ratio:

```python
optimal_calibration = min(1.0, accuracy / avg_confidence)
```

If confidence > accuracy → method is overconfident → reduce calibration
If confidence < accuracy → method is underconfident → increase calibration

---

## 🎯 Benefits

1. **Continuous Improvement**: System gets better over time as it learns from users
2. **Automatic**: No manual intervention needed
3. **Adaptive**: Responds to changes in image types, AI model improvements, etc.
4. **Conservative**: Uses learning rate to prevent drastic changes
5. **Data-Driven**: Makes decisions based on actual performance data

---

## 📝 Monitoring

### Check System Status

The adaptive learning system automatically logs:
- When updates are triggered
- Current method performance
- Weight changes
- Update timestamps

View logs to monitor learning progress.

### Expected Improvements

Over time, you should see:
- Overall accuracy improving as better methods get higher weights
- Confidence scores becoming more calibrated
- System adapting to new types of images

---

## 🔄 Future Enhancements

Potential future improvements:
1. **Threshold Adjustment**: Automatically adjust method-specific thresholds
2. **Factor Weight Tuning**: Adjust individual factor weights in Method 2
3. **Online Learning**: Fine-tune deep learning models based on feedback
4. **A/B Testing**: Test different weight configurations
5. **Performance Tracking**: Long-term tracking of accuracy trends

---

## ⚠️ Notes

- The system requires at least 20 feedback samples before updating
- Updates happen at most once every 24 hours (configurable)
- Learning rate prevents sudden changes (conservative by default)
- Failed learning updates don't affect feedback submission
- The system is designed to be safe and stable

---

## 📚 Related Files

- `detector/adaptive_learning_service.py` - Main learning service
- `detector/three_method_detection_service.py` - Detection service (holds weights)
- `detector/management/commands/adaptive_learn.py` - Management command
- `adaptive_learning_config.json` - Configuration file
- `LATEST_ANALYSIS_REPORT.md` - Latest performance analysis

---

**The system is now active and learning from every piece of feedback you provide!** 🎓✨

