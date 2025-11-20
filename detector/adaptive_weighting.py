"""
IMPROVEMENT 6: Adaptive Weighting System
Automatically adjusts method weights based on real-world performance
"""

import json
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AdaptiveWeightingSystem:
    """
    Tracks method performance and automatically adjusts weights
    """
    
    def __init__(self, weights_config_path='method_weights_config.json'):
        self.weights_config_path = weights_config_path
        self.update_interval = 100  # Update weights every 100 detections
        self.detection_count = 0
        self.performance_history = {}
    
    def record_detection(self, method_id: str, correct: bool, confidence: float):
        """Record a detection result for a method"""
        if method_id not in self.performance_history:
            self.performance_history[method_id] = {
                'total': 0,
                'correct': 0,
                'confidence_sum': 0.0
            }
        
        hist = self.performance_history[method_id]
        hist['total'] += 1
        if correct:
            hist['correct'] += 1
        hist['confidence_sum'] += confidence
        
        self.detection_count += 1
    
    def should_update_weights(self) -> bool:
        """Check if it's time to update weights"""
        return self.detection_count >= self.update_interval
    
    def calculate_adaptive_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate new weights based on performance history
        
        Returns:
            Updated weights dictionary
        """
        if not self.performance_history:
            return current_weights
        
        new_weights = {}
        total_accuracy = 0.0
        
        # Calculate accuracy for each method
        accuracies = {}
        for method_id, hist in self.performance_history.items():
            if hist['total'] > 0:
                accuracy = hist['correct'] / hist['total']
                accuracies[method_id] = accuracy
                total_accuracy += accuracy
        
        if total_accuracy == 0:
            return current_weights
        
        # Redistribute weights proportionally to accuracy
        for method_id in current_weights.keys():
            if method_id in accuracies:
                # Weight = (method_accuracy / total_accuracy) * smoothing
                # Smoothing: blend with current weight to avoid drastic changes
                adaptive_weight = accuracies[method_id] / total_accuracy
                current_weight = current_weights[method_id]
                
                # Blend: 70% adaptive, 30% current (smooth transition)
                new_weights[method_id] = 0.7 * adaptive_weight + 0.3 * current_weight
            else:
                # Keep current weight if no performance data
                new_weights[method_id] = current_weights[method_id]
        
        # Normalize to sum to 1.0
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            new_weights = {k: v/weight_sum for k, v in new_weights.items()}
        
        logger.info("🔄 Adaptive weights calculated:")
        for method_id, weight in new_weights.items():
            old_weight = current_weights.get(method_id, 0)
            acc = accuracies.get(method_id, 0)
            logger.info(f"  {method_id}: {old_weight*100:.1f}% → {weight*100:.1f}% (accuracy: {acc*100:.1f}%)")
        
        return new_weights
    
    def save_weights(self, weights: Dict[str, float], calibration: Dict[str, float]):
        """Save updated weights to config file"""
        try:
            config = {
                'weights': weights,
                'calibration': calibration,
                'detection_count': self.detection_count,
                'performance_history': self.performance_history
            }
            
            with open(self.weights_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✓ Saved adaptive weights to {self.weights_config_path}")
        except Exception as e:
            logger.error(f"Failed to save adaptive weights: {e}")
    
    def reset_counter(self):
        """Reset detection counter after weight update"""
        self.detection_count = 0
        # Keep performance history for next iteration

