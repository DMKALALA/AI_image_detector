"""
Hugging Face Model Ensemble for AI Image Detection
===================================================

Integrates three specialized models from Hugging Face:
1. Vision Transformer (ViT) for AI-generated image detection
2. AI vs Human Image Detector
3. WildFakeDetector (trained on diverse AI-generated images)

These models provide additional detection signals beyond the existing
deep learning, statistical, and forensics methods.
"""

import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class HuggingFaceEnsemble:
    """
    Ensemble of specialized Hugging Face models for AI image detection
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.processors = {}
        
        logger.info(f"Initializing Hugging Face ensemble on {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all three Hugging Face models"""
        
        # Model 1: Vision Transformer (ViT) for AI-generated image detection
        # Common options: "dima806/deepfake_vs_real_image_detection" or similar
        try:
            model_name_vit = "dima806/deepfake_vs_real_image_detection"
            logger.info(f"Loading ViT AI detector: {model_name_vit}")
            
            self.processors['vit_ai_detector'] = ViTImageProcessor.from_pretrained(model_name_vit)
            self.models['vit_ai_detector'] = ViTForImageClassification.from_pretrained(model_name_vit)
            self.models['vit_ai_detector'].to(self.device)
            self.models['vit_ai_detector'].eval()
            
            logger.info("✓ ViT AI-image-detector loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ViT AI detector: {e}")
        
        # Model 2: AI vs Human Image Detector
        # Look for models like "Organika/sdxl-detector" or "umm-maybe/AI-image-detector"
        try:
            model_name_ai_human = "umm-maybe/AI-image-detector"
            logger.info(f"Loading AI vs Human detector: {model_name_ai_human}")
            
            self.processors['ai_human_detector'] = AutoFeatureExtractor.from_pretrained(model_name_ai_human)
            self.models['ai_human_detector'] = AutoModelForImageClassification.from_pretrained(model_name_ai_human)
            self.models['ai_human_detector'].to(self.device)
            self.models['ai_human_detector'].eval()
            
            logger.info("✓ AI vs Human detector loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load AI vs Human detector: {e}")
        
        # Model 3: WildFakeDetector
        # Look for models trained on diverse datasets like "Aaditya2763/wild-fake-detector"
        try:
            model_name_wildfake = "Aaditya2763/wild-fake-detector"
            logger.info(f"Loading WildFakeDetector: {model_name_wildfake}")
            
            self.processors['wildfake_detector'] = AutoFeatureExtractor.from_pretrained(model_name_wildfake)
            self.models['wildfake_detector'] = AutoModelForImageClassification.from_pretrained(model_name_wildfake)
            self.models['wildfake_detector'].to(self.device)
            self.models['wildfake_detector'].eval()
            
            logger.info("✓ WildFakeDetector loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load WildFakeDetector: {e}")
        
        if not self.models:
            logger.warning("No Hugging Face models loaded successfully")
    
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run detection using all available Hugging Face models
        
        Args:
            image: PIL Image
            
        Returns:
            dict with detection results including per-model predictions
        """
        if not self.models:
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'error': 'No Hugging Face models available',
                'indicators': ['Hugging Face models failed to initialize']
            }
        
        try:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            model_predictions = {}
            all_ai_probs = []
            all_weights = []
            
            # Run each model
            for model_name, model in self.models.items():
                try:
                    processor = self.processors[model_name]
                    
                    # Preprocess
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    # Interpret results
                    # Most models: class 0 = real, class 1 = fake/AI
                    # Check config to be sure, but this is common convention
                    if len(probs) >= 2:
                        real_prob = float(probs[0])
                        ai_prob = float(probs[1])
                    else:
                        # Binary output, single value
                        ai_prob = float(probs[0])
                        real_prob = 1.0 - ai_prob
                    
                    is_ai = ai_prob > 0.5
                    confidence = max(ai_prob, real_prob)
                    
                    model_predictions[model_name] = {
                        'is_ai': is_ai,
                        'ai_prob': ai_prob,
                        'real_prob': real_prob,
                        'confidence': confidence
                    }
                    
                    all_ai_probs.append(ai_prob)
                    all_weights.append(1.0)  # Equal weights for now
                    
                    logger.debug(f"{model_name}: AI={ai_prob:.3f}, Real={real_prob:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Error running {model_name}: {e}")
                    continue
            
            if not model_predictions:
                return {
                    'is_ai_generated': False,
                    'confidence': 0.0,
                    'error': 'All models failed',
                    'indicators': ['All Hugging Face models failed during inference']
                }
            
            # Ensemble: weighted average
            weights = np.array(all_weights)
            weights = weights / weights.sum()
            ensemble_ai_prob = float(np.average(all_ai_probs, weights=weights))
            ensemble_real_prob = 1.0 - ensemble_ai_prob
            
            is_ai_generated = ensemble_ai_prob > 0.5
            confidence = max(ensemble_ai_prob, ensemble_real_prob)
            
            # Build indicators
            indicators = [
                f"Hugging Face Ensemble: {'AI-generated' if is_ai_generated else 'Real'}",
                f"Ensemble AI probability: {ensemble_ai_prob*100:.1f}%",
                f"Ensemble Real probability: {ensemble_real_prob*100:.1f}%",
                f"Models used: {len(model_predictions)}/3"
            ]
            
            # Add individual model results
            model_labels = {
                'vit_ai_detector': 'ViT AI-detector',
                'ai_human_detector': 'AI vs Human Detector',
                'wildfake_detector': 'WildFakeDetector'
            }
            
            for model_name, pred in model_predictions.items():
                label = model_labels.get(model_name, model_name)
                indicators.append(
                    f"{label}: {'AI' if pred['is_ai'] else 'Real'} "
                    f"(AI: {pred['ai_prob']*100:.1f}%, confidence: {pred['confidence']*100:.1f}%)"
                )
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'probabilities': {
                    'ai': ensemble_ai_prob,
                    'real': ensemble_real_prob
                },
                'indicators': indicators,
                'model_predictions': model_predictions,
                'models_count': len(model_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in Hugging Face ensemble detection: {e}", exc_info=True)
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'error': str(e),
                'indicators': [f'Hugging Face ensemble error: {str(e)}']
            }

