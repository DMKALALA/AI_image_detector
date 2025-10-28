# Factor-Level Analysis & Improvements

## 📊 Detailed Analysis Results (56 Recent Uploads)

### Method Performance
| Method | Accuracy | Status | Action Taken |
|--------|----------|--------|--------------|
| **Method 1: Deep Learning** | 37.0% (17/46) | ❌ POOR | Weight reduced to 10% |
| **Method 2: Statistical** | **63.0%** (29/46) | ✅ GOOD | Weight increased to 75%, factors boosted |
| **Method 3: Metadata** | 37.0% (17/46) | ❌ POOR | Weight reduced to 15%, weak factors removed |

---

## 🎯 Factor-Level Analysis

### Method 2 Factors (Statistical Pattern Analysis)

| Factor | Accuracy | Previous Weight | New Weight | Change |
|--------|----------|----------------|------------|--------|
| **low_edge_density** | **65.8%** (25/38) | 0.25 | **0.375** | ✅ **+50% Boost** (1.5x) |
| regular_frequency_pattern | 63.0% (29/46) | 0.15 | 0.15 | No change |
| color_banding | 59.1% (13/22) | 0.15 | 0.15 | No change |
| high_edge_density | - | 0.15 | 0.12 | ✅ Reduced (0.8x) - causes false positives |
| uniform_brightness | - | 0.20 | 0.18 | ✅ Slightly reduced (0.9x) |

**Key Finding**: `low_edge_density` is the most accurate factor at 65.8% - it now contributes **50% more** to the AI detection score.

### Method 3 Factors (Metadata & Heuristic Analysis)

| Factor | Accuracy | Previous Weight | New Weight | Change |
|--------|----------|----------------|------------|--------|
| **no_factors** | **64.7%** (11/17) | N/A | N/A | ✅ Default to "Real" when no factors |
| common_ai_resolution | 15.0% (3/20) | 0.15 | **0.05** | ✅ **-67% Reduced** (weak indicator) |
| common_ai_size | 29.4% (5/17) | 0.05 | **REMOVED** | ✅ **Eliminated** (too inaccurate) |
| square_aspect_ratio | 0.0% (0/1) | 0.10 | **REMOVED** | ✅ **Eliminated** (0% accuracy!) |

**Key Findings**:
- "no_factors" case is **64.7% accurate** - Method 3 now defaults to "Real" when no indicators are found
- Weak factors significantly reduced or removed
- Strong factors (EXIF metadata, filename patterns) remain at full weight

---

## 🔧 Implemented Improvements

### 1. Method Weight Adjustments

**Previous Weights**:
- Method 1: 20%
- Method 2: 60%
- Method 3: 20%

**New Weights** (based on 63% vs 37% accuracy):
- Method 1: **10%** (reduced - consistently poor)
- Method 2: **75%** (increased - only reliable method)
- Method 3: **15%** (reduced - consistently poor)

**Reasoning**: Method 2 is the only method achieving acceptable accuracy (63%). Methods 1 & 3 at 37% are essentially worse than random guessing, so their influence is minimized.

### 2. Confidence Calibration Updates

**New Calibration Factors**:
- Method 1: **0.3** (was 0.4) - heavily penalize overconfidence
- Method 2: **0.95** (was 0.9) - slight reduction only, good calibration
- Method 3: **0.4** (was 0.5) - heavily penalize overconfidence

**Reasoning**: Methods 1 & 3 are overconfident (99.9% and 91.4% confidence but wrong 63% of the time). Method 2 has better calibration (67.4% avg confidence, 63% accuracy).

### 3. Method 2 Factor Weights

**New Factor Weight System**:
```python
method_2_factor_weights = {
    'low_color_variation': 1.0,        # Standard
    'low_edge_density': 1.5,           # BOOSTED - 65.8% accuracy
    'high_edge_density': 0.8,          # Reduced - causes false positives
    'uniform_texture': 1.0,            # Standard
    'uniform_brightness': 0.9,         # Slightly reduced
    'color_banding': 1.0,              # Standard (59.1% accuracy)
    'regular_frequency_pattern': 1.0    # Standard (63.0% accuracy)
}
```

**Impact**: When `low_edge_density` is detected, it now contributes **0.375** to the AI score (was 0.25), making it the strongest single indicator.

### 4. Method 3 Overhaul

**Removed Factors**:
- ❌ `common_ai_size` - Only 29.4% accuracy
- ❌ `square_aspect_ratio` - 0% accuracy!

**Reduced Factors**:
- `common_ai_resolution`: 0.15 → **0.05** (15% accuracy - very weak)
- `unusually_small_file`: 0.10 → **0.05**

**Improved Threshold Logic**:
- **Strong factors** (EXIF metadata, filename patterns): Threshold = 0.35
- **Weak factors only**: Threshold = 0.50 (requires very high score)
- **No factors**: Default to "Real" (64.7% accurate)

**Confidence Reduction**:
- If only weak factors detected: Confidence reduced by 30%

**Reasoning**: Method 3's "no_factors" case is more accurate (64.7%) than any single weak factor. The method now prioritizes strong, reliable indicators and defaults to "Real" when uncertain.

---

## 📈 Expected Improvements

### Accuracy Improvements

**Before Changes**:
- Final accuracy: 45.7%
- Method 2: 63.0% (but only 60% weight)
- Methods 1 & 3: 37% (but 20% weight each)

**After Changes** (Expected):
- Final accuracy: **~55-60%** (target)
- Method 2: 63% with **75% weight** (dominant influence)
- Methods 1 & 3: 37% with minimal weight (10% + 15%)

### Factor Performance

- **low_edge_density**: Now contributes 50% more weight when detected
- **Weak Method 3 factors**: Removed or significantly reduced
- **Strong Method 3 factors**: Retained and better calibrated

### Confidence Calibration

- **Method 1 & 3**: Heavily penalized for overconfidence
- **Method 2**: Better calibration retained
- **Method 3**: Confidence reduced when only weak factors present

---

## 🎯 Summary of Changes

1. ✅ **Method weights rebalanced**: Method 2 = 75%, Methods 1 & 3 = 10% + 15%
2. ✅ **low_edge_density boosted**: 1.5x weight (65.8% accuracy)
3. ✅ **Weak Method 3 factors removed/reduced**: common_ai_size removed, common_ai_resolution reduced
4. ✅ **Method 3 threshold logic improved**: Requires strong factors or very high weak-factor scores
5. ✅ **Method 3 defaults to "Real"**: When no factors detected (64.7% accurate)
6. ✅ **Confidence calibration tightened**: Methods 1 & 3 heavily penalized

---

## 📊 Monitoring Recommendations

### Track These Metrics

1. **Overall Accuracy**: Should improve from 45.7% to ~55-60%
2. **Method 2 Performance**: Should maintain 63%+ with greater influence
3. **Method 1 & 3 Impact**: Should have minimal negative impact (low weights)
4. **Factor Effectiveness**: Monitor if `low_edge_density` boost improves results

### Re-Analyze Regularly

Run factor analysis periodically:
```bash
python manage.py analyze_detection_factors --limit 60
```

Look for:
- Any factor consistently achieving 65%+ accuracy → boost weight
- Any factor consistently underperforming → reduce or remove
- Method accuracy trends → adjust method weights accordingly

---

## 🔍 Method 1 & 3 Replacement Consideration

**Current Status**: Both methods at 37% accuracy (worse than random)

### Method 1 (Deep Learning)
- **Issue**: Consistently predicts "Real" when images are AI (29 errors)
- **Recommendation**: 
  - Consider retraining with more diverse AI examples
  - Lower the decision threshold for AI detection
  - Or disable entirely if performance doesn't improve

### Method 3 (Metadata)
- **Issue**: Most factors are inaccurate (15-29% accuracy)
- **Improvement Applied**: 
  - Removed weak factors (common_ai_size, square_aspect_ratio)
  - Reduced weight of weak factors (common_ai_resolution)
  - Improved threshold logic to prioritize strong factors
  - Default to "Real" when no factors (64.7% accurate)

**Status**: Method 3 is now more conservative and should have less negative impact. Still recommended to keep weight low (15%) until factor accuracy improves.

---

## ✅ Next Steps

1. **Test with new uploads**: Verify improvements in real-world detection
2. **Monitor factor accuracy**: Track which factors continue to perform well
3. **Consider Method 1 replacement**: If accuracy doesn't improve, replace with alternative approach
4. **Continued refinement**: Use factor analysis to fine-tune weights continuously

The system is now optimized based on detailed factor-level analysis, focusing on the most reliable detection methods and indicators!

