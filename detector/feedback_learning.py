"""
Feedback-Based Learning System
===============================

Uses user feedback to:
1. Track image hashes to recognize re-uploads
2. Remember correct predictions for specific images
3. Adjust confidence when same image is re-uploaded
4. Learn from mistakes to avoid repeating errors
"""

import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FeedbackLearningService:
    """
    Service that learns from user feedback to improve predictions
    """
    
    def __init__(self, feedback_db_path='feedback_memory.json'):
        self.feedback_db_path = feedback_db_path
        self.feedback_memory = self._load_feedback_memory()
    
    def _load_feedback_memory(self) -> Dict:
        """Load feedback memory from disk"""
        if os.path.exists(self.feedback_db_path):
            try:
                with open(self.feedback_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load feedback memory: {e}")
                return {}
        return {}
    
    def _save_feedback_memory(self):
        """Save feedback memory to disk"""
        try:
            with open(self.feedback_db_path, 'w') as f:
                json.dump(self.feedback_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save feedback memory: {e}")
    
    def compute_image_hash(self, image_path: str) -> str:
        """
        Compute perceptual hash of image for re-upload detection
        Uses MD5 of image content (simple but effective)
        """
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing image hash: {e}")
            return None
    
    def check_previous_feedback(self, image_hash: str) -> Optional[Dict]:
        """
        Check if we've seen this image before and have feedback
        
        Returns:
            dict with previous result and feedback if found, None otherwise
        """
        if image_hash in self.feedback_memory:
            memory = self.feedback_memory[image_hash]
            return {
                'seen_before': True,
                'previous_prediction': memory.get('prediction'),
                'user_feedback': memory.get('feedback'),
                'correct_answer': memory.get('correct_answer'),
                'times_seen': memory.get('times_seen', 1),
                'confidence_adjustment': self._calculate_confidence_adjustment(memory)
            }
        return None
    
    def _calculate_confidence_adjustment(self, memory: Dict) -> float:
        """
        Calculate confidence boost/penalty based on feedback history
        
        Returns:
            Multiplier for confidence (1.0 = no change, >1.0 = boost, <1.0 = penalty)
        """
        feedback = memory.get('feedback')
        
        if feedback == 'correct':
            # User confirmed our prediction was right - boost confidence
            return 1.15  # +15% confidence boost
        elif feedback == 'incorrect':
            # User said we were wrong - reduce confidence significantly
            return 0.60  # -40% confidence penalty
        else:
            # Unsure or no feedback
            return 1.0
    
    def record_feedback(self, image_hash: str, prediction: bool, 
                       confidence: float, user_feedback: str,
                       correct_answer: Optional[bool] = None):
        """
        Record user feedback for an image
        
        Args:
            image_hash: Hash of the image
            prediction: Our prediction (True = AI, False = Real)
            confidence: Our confidence score
            user_feedback: 'correct', 'incorrect', or 'unsure'
            correct_answer: The actual answer if feedback is 'incorrect'
        """
        if image_hash not in self.feedback_memory:
            self.feedback_memory[image_hash] = {
                'times_seen': 0,
                'predictions': [],
                'feedbacks': []
            }
        
        memory = self.feedback_memory[image_hash]
        memory['times_seen'] = memory.get('times_seen', 0) + 1
        memory['prediction'] = prediction
        memory['confidence'] = confidence
        memory['feedback'] = user_feedback
        
        # Track prediction history
        memory['predictions'].append({
            'prediction': prediction,
            'confidence': confidence,
            'feedback': user_feedback
        })
        
        # If user said we were wrong, record the correct answer
        if user_feedback == 'incorrect' and correct_answer is not None:
            memory['correct_answer'] = correct_answer
        elif user_feedback == 'correct':
            memory['correct_answer'] = prediction
        
        # Keep only last 10 predictions
        if len(memory['predictions']) > 10:
            memory['predictions'] = memory['predictions'][-10:]
        
        self._save_feedback_memory()
        logger.info(f"Recorded feedback for image {image_hash[:8]}: {user_feedback}")
    
    def apply_learning_adjustment(self, image_path: str, 
                                 prediction_result: Dict) -> Dict:
        """
        Apply learning adjustments based on feedback history
        
        Args:
            image_path: Path to the image being analyzed
            prediction_result: Current prediction result
            
        Returns:
            Adjusted prediction result with feedback-based modifications
        """
        image_hash = self.compute_image_hash(image_path)
        if not image_hash:
            return prediction_result
        
        previous = self.check_previous_feedback(image_hash)
        
        if previous and previous['seen_before']:
            # We've seen this image before!
            times_seen = previous['times_seen']
            user_feedback = previous.get('user_feedback')
            correct_answer = previous.get('correct_answer')
            
            indicators = prediction_result.get('indicators', [])
            
            # Add re-upload indicator
            indicators.insert(0, f"🔄 Image seen {times_seen} time(s) before")
            
            # If we know the correct answer from previous feedback
            if correct_answer is not None and user_feedback == 'incorrect':
                # Override prediction with learned answer
                indicators.insert(1, f"📚 Learned from feedback: This is {'AI' if correct_answer else 'Real'}")
                prediction_result['is_ai_generated'] = correct_answer
                prediction_result['confidence'] = min(0.95, prediction_result['confidence'] * 1.2)
                indicators.insert(2, "Confidence boosted based on feedback history")
            
            elif user_feedback == 'correct':
                # User confirmed we were right before - boost confidence
                adjustment = previous['confidence_adjustment']
                old_conf = prediction_result['confidence']
                prediction_result['confidence'] = min(0.98, old_conf * adjustment)
                indicators.insert(1, f"✅ Previously confirmed correct (confidence +{(adjustment-1)*100:.0f}%)")
            
            elif user_feedback == 'incorrect':
                # We were wrong before - be more cautious
                adjustment = previous['confidence_adjustment']
                old_conf = prediction_result['confidence']
                prediction_result['confidence'] = old_conf * adjustment
                indicators.insert(1, f"⚠️ Previously incorrect (confidence reduced {(1-adjustment)*100:.0f}%)")
            
            prediction_result['indicators'] = indicators
            prediction_result['feedback_adjusted'] = True
            prediction_result['image_hash'] = image_hash
        
        else:
            # First time seeing this image
            prediction_result['feedback_adjusted'] = False
            prediction_result['image_hash'] = image_hash
        
        return prediction_result


# IMPROVEMENT 7: Generalized Feedback Learning
class GeneralizedFeedbackLearning:
    """
    Learn from feedback beyond just re-uploads
    Builds a profile of what makes images AI vs Real based on user corrections
    """
    
    def __init__(self):
        self.learning_db_path = 'generalized_feedback_patterns.json'
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> dict:
        """Load learned patterns from disk"""
        if os.path.exists(self.learning_db_path):
            try:
                with open(self.learning_db_path, 'r') as f:
                    return json.load(f)
            except:
                return {'ai_indicators': {}, 'real_indicators': {}}
        return {'ai_indicators': {}, 'real_indicators': {}}
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        try:
            with open(self.learning_db_path, 'w') as f:
                json.dump(self.patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def learn_from_correction(self, indicators: list, was_prediction: bool, 
                             actual_answer: bool):
        """
        Learn what indicators correlate with AI vs Real
        When user corrects us, update indicator weights
        """
        if was_prediction == actual_answer:
            # We were right, reinforce these indicators
            target = 'ai_indicators' if actual_answer else 'real_indicators'
        else:
            # We were wrong, these indicators mislead us
            # Record in opposite category with negative weight
            target = 'real_indicators' if was_prediction else 'ai_indicators'
        
        for indicator in indicators:
            # Extract key phrases from indicator text
            phrases = self._extract_key_phrases(indicator)
            
            for phrase in phrases:
                if phrase not in self.patterns[target]:
                    self.patterns[target][phrase] = {'count': 0, 'weight': 0.0}
                
                pattern = self.patterns[target][phrase]
                pattern['count'] += 1
                
                # Adjust weight: positive for correct, negative for incorrect
                if was_prediction == actual_answer:
                    pattern['weight'] += 0.1  # Reinforce
                else:
                    pattern['weight'] -= 0.15  # Penalize misleading indicators
        
        self._save_patterns()
        logger.info(f"📚 Learned from correction: {actual_answer} (was {was_prediction})")
    
    def _extract_key_phrases(self, indicator: str) -> list:
        """Extract meaningful phrases from indicator text"""
        # Simple extraction: look for key technical terms
        phrases = []
        keywords = [
            'spectral', 'texture', 'color', 'edge', 'entropy', 'pattern',
            'frequency', 'noise', 'compression', 'artifact', 'metadata',
            'uniform', 'variation', 'distribution', 'symmetry'
        ]
        
        indicator_lower = indicator.lower()
        for keyword in keywords:
            if keyword in indicator_lower:
                phrases.append(keyword)
        
        return phrases
    
    def adjust_confidence_from_patterns(self, indicators: list, 
                                       prediction: bool, confidence: float) -> float:
        """
        Adjust confidence based on learned patterns
        If indicators match learned AI patterns, boost confidence for AI prediction
        If indicators match learned Real patterns, boost confidence for Real prediction
        """
        ai_score = 0.0
        real_score = 0.0
        
        # Check each indicator against learned patterns
        for indicator in indicators:
            phrases = self._extract_key_phrases(indicator)
            
            for phrase in phrases:
                if phrase in self.patterns['ai_indicators']:
                    ai_score += self.patterns['ai_indicators'][phrase]['weight']
                if phrase in self.patterns['real_indicators']:
                    real_score += self.patterns['real_indicators'][phrase]['weight']
        
        # Adjust confidence based on pattern alignment
        if prediction:  # Predicting AI
            if ai_score > real_score:
                adjustment = min(0.15, (ai_score - real_score) * 0.05)
                confidence = min(0.95, confidence * (1 + adjustment))
            else:
                adjustment = min(0.15, (real_score - ai_score) * 0.05)
                confidence = confidence * (1 - adjustment)
        else:  # Predicting Real
            if real_score > ai_score:
                adjustment = min(0.15, (real_score - ai_score) * 0.05)
                confidence = min(0.95, confidence * (1 + adjustment))
            else:
                adjustment = min(0.15, (ai_score - real_score) * 0.05)
                confidence = confidence * (1 - adjustment)
        
        return confidence


# Global instances
_feedback_service = None
_generalized_learning = None

def get_feedback_service():
    """Get or create the feedback learning service instance"""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackLearningService()
    return _feedback_service

def get_generalized_learning():
    """Get or create the generalized learning instance"""
    global _generalized_learning
    if _generalized_learning is None:
        _generalized_learning = GeneralizedFeedbackLearning()
    return _generalized_learning

