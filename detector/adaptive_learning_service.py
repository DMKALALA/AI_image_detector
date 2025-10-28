"""
Adaptive Learning Service
Automatically updates detection service weights and thresholds based on user feedback
Implements continuous learning from feedback to improve accuracy over time
"""

import logging
from django.utils import timezone
from datetime import timedelta
from detector.models import ImageUpload
from detector.three_method_detection_service import three_method_detection_service
import json
import os

logger = logging.getLogger(__name__)

class AdaptiveLearningService:
    """
    Service that continuously learns from user feedback and adapts
    the detection service weights and parameters automatically
    """
    
    def __init__(self):
        self.learning_config_path = 'adaptive_learning_config.json'
        self.load_config()
        logger.info("Adaptive Learning Service initialized")
    
    def load_config(self):
        """Load learning configuration"""
        default_config = {
            'auto_update_enabled': True,
            'update_interval_hours': 24,  # Update weights every 24 hours
            'min_feedback_samples': 20,  # Minimum feedback samples before updating
            'learning_rate': 0.1,  # How aggressively to update weights
            'last_update': None
        }
        
        if os.path.exists(self.learning_config_path):
            try:
                with open(self.learning_config_path, 'r') as f:
                    self.config = json.load(f)
                    # Ensure all keys exist
                    for key, value in default_config.items():
                        if key not in self.config:
                            self.config[key] = value
            except Exception as e:
                logger.warning(f"Error loading learning config: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save learning configuration"""
        try:
            with open(self.learning_config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning config: {e}")
    
    def should_update(self):
        """Check if it's time to update weights"""
        if not self.config.get('auto_update_enabled', True):
            return False
        
        last_update = self.config.get('last_update')
        if not last_update:
            return True
        
        try:
            from datetime import datetime
            last_update_time = datetime.fromisoformat(last_update)
            hours_since_update = (timezone.now() - last_update_time).total_seconds() / 3600
            return hours_since_update >= self.config.get('update_interval_hours', 24)
        except:
            return True
    
    def analyze_feedback(self, limit=100):
        """
        Analyze recent feedback to calculate current method performance
        Returns accuracy and performance metrics for each method
        """
        try:
            # Get recent uploads with feedback
            recent_uploads = ImageUpload.objects.filter(
                user_feedback__in=['correct', 'incorrect']
            ).order_by('-uploaded_at')[:limit]
            
            if len(recent_uploads) < self.config.get('min_feedback_samples', 20):
                logger.info(f"Not enough feedback samples ({len(recent_uploads)}), need {self.config.get('min_feedback_samples', 20)}")
                return None
            
            # Analyze each method's performance
            method_stats = {
                'method_1': {'correct': 0, 'total': 0, 'confidences': []},
                'method_2': {'correct': 0, 'total': 0, 'confidences': []},
                'method_3': {'correct': 0, 'total': 0, 'confidences': []}
            }
            
            for upload in recent_uploads:
                # Determine actual label from feedback
                if upload.user_feedback == 'correct':
                    actual_is_ai = upload.is_ai_generated
                elif upload.user_feedback == 'incorrect':
                    actual_is_ai = not upload.is_ai_generated
                else:
                    continue
                
                # Extract method results from analysis_details
                if upload.analysis_details and 'method_comparison' in upload.analysis_details:
                    comparison = upload.analysis_details['method_comparison']
                    
                    for method_key in ['method_1', 'method_2', 'method_3']:
                        if method_key in comparison:
                            method_result = comparison[method_key]
                            predicted_is_ai = method_result.get('is_ai_generated', False)
                            confidence = method_result.get('confidence', 0.0)
                            
                            method_stats[method_key]['total'] += 1
                            method_stats[method_key]['confidences'].append(confidence)
                            
                            if predicted_is_ai == actual_is_ai:
                                method_stats[method_key]['correct'] += 1
            
            # Calculate accuracies
            results = {}
            for method_key, stats in method_stats.items():
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total']
                    avg_confidence = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0.0
                    results[method_key] = {
                        'accuracy': accuracy,
                        'correct': stats['correct'],
                        'total': stats['total'],
                        'avg_confidence': avg_confidence
                    }
                else:
                    results[method_key] = {
                        'accuracy': 0.0,
                        'correct': 0,
                        'total': 0,
                        'avg_confidence': 0.0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}", exc_info=True)
            return None
    
    def calculate_optimal_weights(self, method_performances):
        """
        Calculate optimal method weights based on accuracy
        Uses softmax-like function to convert accuracies to weights
        """
        if not method_performances:
            return None
        
        # Extract accuracies
        accuracies = {
            method: perf['accuracy'] for method, perf in method_performances.items()
        }
        
        # Normalize accuracies (add small value to avoid zero)
        total_accuracy = sum(accuracies.values()) + 0.01
        normalized = {method: acc / total_accuracy for method, acc in accuracies.items()}
        
        # Apply exponential to emphasize differences (softmax-like)
        import math
        exp_accuracies = {method: math.exp(acc * 5) for method, acc in normalized.items()}
        total_exp = sum(exp_accuracies.values())
        
        # Calculate weights
        weights = {method: exp_acc / total_exp for method, exp_acc in exp_accuracies.items()}
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {method: w / total for method, w in weights.items()}
        
        return weights
    
    def update_service_weights(self, new_weights):
        """
        Update the detection service weights adaptively
        Uses learning rate to blend old and new weights
        """
        try:
            service = three_method_detection_service
            learning_rate = self.config.get('learning_rate', 0.1)
            
            # Get current weights
            current_weights = service.method_accuracy_weights.copy()
            
            # Blend old and new weights
            updated_weights = {}
            for method_key in ['method_1', 'method_2', 'method_3']:
                old_weight = current_weights.get(method_key, 0.33)
                new_weight = new_weights.get(method_key, 0.33)
                
                # Exponential moving average
                updated_weight = (1 - learning_rate) * old_weight + learning_rate * new_weight
                updated_weights[method_key] = updated_weight
            
            # Normalize to sum to 1.0
            total = sum(updated_weights.values())
            if total > 0:
                updated_weights = {method: w / total for method, w in updated_weights.items()}
            
            # Update service
            service.method_accuracy_weights = updated_weights
            
            # Save weights to config file for persistence (calibration saved separately)
            weights_config = {
                'weights': updated_weights,
                'calibration': service.confidence_calibration  # Use current calibration, will be updated separately
            }
            weights_config_path = 'method_weights_config.json'
            try:
                with open(weights_config_path, 'w') as f:
                    json.dump(weights_config, f, indent=2)
                logger.info(f"Saved method weights to {weights_config_path}")
            except Exception as e:
                logger.warning(f"Could not save weights: {e}")
            
            logger.info(f"Updated method weights: {updated_weights}")
            return updated_weights
            
        except Exception as e:
            logger.error(f"Error updating service weights: {e}", exc_info=True)
            return None
    
    def update_confidence_calibration(self, method_performances):
        """
        Update confidence calibration based on actual performance
        If a method is overconfident (high confidence, low accuracy), reduce calibration
        """
        try:
            service = three_method_detection_service
            learning_rate = self.config.get('learning_rate', 0.1)
            
            current_calibration = service.confidence_calibration.copy()
            updated_calibration = {}
            
            for method_key in ['method_1', 'method_2', 'method_3']:
                if method_key in method_performances:
                    perf = method_performances[method_key]
                    accuracy = perf['accuracy']
                    avg_confidence = perf['avg_confidence']
                    
                    # If confidence > accuracy, method is overconfident
                    # Calibration should be adjusted based on accuracy/confidence ratio
                    if avg_confidence > 0:
                        optimal_calibration = min(1.0, accuracy / avg_confidence) if avg_confidence > 0 else 0.5
                    else:
                        optimal_calibration = 0.5
                    
                    # Blend with current calibration
                    current = current_calibration.get(method_key, 1.0)
                    updated = (1 - learning_rate) * current + learning_rate * optimal_calibration
                    updated_calibration[method_key] = max(0.1, min(1.0, updated))  # Clamp between 0.1 and 1.0
                else:
                    updated_calibration[method_key] = current_calibration.get(method_key, 1.0)
            
            service.confidence_calibration = updated_calibration
            
            # Save calibration along with weights
            weights_config_path = 'method_weights_config.json'
            if os.path.exists(weights_config_path):
                try:
                    with open(weights_config_path, 'r') as f:
                        weights_config = json.load(f)
                    weights_config['calibration'] = updated_calibration
                    with open(weights_config_path, 'w') as f:
                        json.dump(weights_config, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not save calibration: {e}")
            
            logger.info(f"Updated confidence calibration: {updated_calibration}")
            return updated_calibration
            
        except Exception as e:
            logger.error(f"Error updating confidence calibration: {e}", exc_info=True)
            return None
    
    def learn_and_update(self):
        """
        Main learning function: analyzes feedback and updates service parameters
        """
        if not self.should_update():
            logger.info("Not time to update yet")
            return False
        
        logger.info("Starting adaptive learning update...")
        
        # Analyze recent feedback
        method_performances = self.analyze_feedback()
        
        if not method_performances:
            logger.info("Not enough feedback data for learning")
            return False
        
        # Log current performance
        logger.info("Current method performance:")
        for method, perf in method_performances.items():
            logger.info(f"  {method}: {perf['accuracy']*100:.1f}% accuracy ({perf['correct']}/{perf['total']})")
        
        # Calculate optimal weights
        optimal_weights = self.calculate_optimal_weights(method_performances)
        
        if optimal_weights:
            # Update weights
            self.update_service_weights(optimal_weights)
            
            # Update confidence calibration
            self.update_confidence_calibration(method_performances)
            
            # Update last update time
            self.config['last_update'] = timezone.now().isoformat()
            self.save_config()
            
            logger.info("✅ Adaptive learning update complete")
            return True
        
        return False
    
    def trigger_learning_on_feedback(self, image_upload):
        """
        Triggered automatically when feedback is submitted
        Checks if enough samples exist and updates if needed
        """
        try:
            # Quick check: count recent feedback
            recent_count = ImageUpload.objects.filter(
                user_feedback__in=['correct', 'incorrect']
            ).count()
            
            min_samples = self.config.get('min_feedback_samples', 20)
            
            if recent_count >= min_samples:
                # Check if it's been a while since last update
                # Or if we have significantly more samples
                if self.should_update():
                    logger.info(f"Triggering adaptive learning ({recent_count} feedback samples)")
                    self.learn_and_update()
            
        except Exception as e:
            logger.error(f"Error in trigger_learning_on_feedback: {e}", exc_info=True)

# Global instance
adaptive_learning_service = AdaptiveLearningService()

