"""
IMPROVEMENT 3: Auto-Retraining Trigger
Automatically triggers model retraining when enough feedback is collected
"""

import os
import subprocess
import logging
from django.db import models
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoRetrainingSystem:
    """
    Monitors feedback count and triggers retraining automatically
    """
    
    def __init__(self, min_feedback_samples=50, retrain_interval=100):
        self.min_feedback_samples = min_feedback_samples
        self.retrain_interval = retrain_interval
        self.last_retrain_count = self._load_last_retrain_count()
    
    def _load_last_retrain_count(self) -> int:
        """Load the count at which we last retrained"""
        marker_file = 'last_retrain_marker.txt'
        if os.path.exists(marker_file):
            try:
                with open(marker_file, 'r') as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0
    
    def _save_retrain_marker(self, count: int):
        """Save current feedback count as retrain marker"""
        try:
            with open('last_retrain_marker.txt', 'w') as f:
                f.write(str(count))
        except Exception as e:
            logger.error(f"Failed to save retrain marker: {e}")
    
    def should_trigger_retrain(self, total_feedback_count: int) -> bool:
        """
        Check if we should trigger retraining
        
        Conditions:
        1. Have minimum feedback samples
        2. New feedback since last retrain >= interval
        """
        if total_feedback_count < self.min_feedback_samples:
            logger.info(f"Not enough feedback for retraining: {total_feedback_count}/{self.min_feedback_samples}")
            return False
        
        new_feedback = total_feedback_count - self.last_retrain_count
        
        if new_feedback >= self.retrain_interval:
            logger.info(f"🔄 Retraining triggered: {new_feedback} new feedback samples since last retrain")
            return True
        
        logger.debug(f"Not triggering retrain: {new_feedback}/{self.retrain_interval} new samples")
        return False
    
    def trigger_retrain(self, total_feedback_count: int):
        """
        Trigger retraining process in background
        """
        try:
            logger.info("🚀 Starting automatic retraining process...")
            
            # Update marker first
            self._save_retrain_marker(total_feedback_count)
            self.last_retrain_count = total_feedback_count
            
            # Run retrain command in background
            # Note: This assumes manage.py is in current directory
            manage_py = Path(__file__).parent.parent / 'manage.py'
            
            # Step 1: Prepare feedback dataset
            subprocess.Popen(
                ['python', str(manage_py), 'retrain_from_feedback', f'--min-feedback={self.min_feedback_samples}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("✓ Retraining process started in background")
            logger.info(f"  Feedback samples: {total_feedback_count}")
            logger.info(f"  New samples since last retrain: {total_feedback_count - self.last_retrain_count}")
            
        except Exception as e:
            logger.error(f"Failed to trigger retrain: {e}")
    
    def check_and_trigger(self):
        """
        Check feedback count and trigger retrain if needed
        Should be called after each feedback submission
        """
        try:
            # Import here to avoid circular dependency
            from detector.models import ImageUpload
            
            # Count feedback
            total_feedback = ImageUpload.objects.exclude(user_feedback='').count()
            
            if self.should_trigger_retrain(total_feedback):
                self.trigger_retrain(total_feedback)
                return True
        except Exception as e:
            logger.error(f"Error in auto-retrain check: {e}")
        
        return False


# Global instance
_auto_retrain_system = None

def get_auto_retrain_system():
    """Get or create auto-retrain system instance"""
    global _auto_retrain_system
    if _auto_retrain_system is None:
        _auto_retrain_system = AutoRetrainingSystem()
    return _auto_retrain_system

