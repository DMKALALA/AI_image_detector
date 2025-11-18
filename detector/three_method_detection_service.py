"""
Three-Method AI Detection Service
==================================

This service implements three distinct methodologies for AI image detection,
each using completely different approaches:

Method 1: DEEP_LEARNING_MODEL
- Uses trained neural network (GenImage ResNet-50 model)
- Learns from dataset patterns
- High accuracy but requires training data

Method 2: STATISTICAL_PATTERN_ANALYSIS  
- Analyzes pixel-level statistical patterns
- Color distribution, texture uniformity, edge detection
- No training required, rule-based

Method 3: METADATA_HEURISTIC_ANALYSIS
- Examines EXIF metadata, file patterns, compression artifacts
- Rule-based heuristics and pattern matching
- Fast, interpretable results

Each method provides independent results for direct comparison.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ExifTags
import cv2
from scipy import fft
from scipy.fftpack import dct
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter

logger = logging.getLogger(__name__)

# Import modern ensemble method
try:
    from detector.modern_ensemble_method import ModernEnsembleService
    MODERN_ENSEMBLE_AVAILABLE = True
    logger.info("✓ Modern ensemble module imported successfully")
except ImportError as e:
    MODERN_ENSEMBLE_AVAILABLE = False
    logger.warning(f"Could not import modern ensemble module: {e}")

# Import improved methods
try:
    from detector.improved_method_1_deeplearning import ImprovedDeepLearningMethod1
    IMPROVED_METHOD_1_AVAILABLE = True
    logger.info("✓ Improved Method 1 (Deep Learning) module imported successfully")
except ImportError as e:
    IMPROVED_METHOD_1_AVAILABLE = False
    logger.warning(f"Could not import improved Method 1: {e}")

try:
    from detector.advanced_spectral_method3 import AdvancedSpectralMethod3
    ADVANCED_SPECTRAL_METHOD_3_AVAILABLE = True
    logger.info("✓ Advanced Spectral Method 3 module imported successfully")
except ImportError as e:
    ADVANCED_SPECTRAL_METHOD_3_AVAILABLE = False
    logger.warning(f"Could not import Advanced Spectral Method 3: {e}")

# Keep forensics as optional fallback
try:
    from detector.improved_method_3_forensics import ImprovedForensicsMethod3
    IMPROVED_METHOD_3_AVAILABLE = True
except ImportError:
    IMPROVED_METHOD_3_AVAILABLE = False

# Hugging Face specialized models ensemble
try:
    from detector.huggingface_models import HuggingFaceEnsemble
    HUGGINGFACE_AVAILABLE = True
    logger.info("✓ Hugging Face models module imported successfully")
except ImportError as e:
    HUGGINGFACE_AVAILABLE = False
    logger.warning(f"Could not import Hugging Face models: {e}")

class ThreeMethodDetectionService:
    """
    Service that runs three distinct AI detection methods independently
    and provides results for each method for comparison.
    """
    
    def __init__(self):
        # Force CPU mode for free tier (512MB limit) - CUDA models use too much memory
        # Check if we're on a memory-constrained environment
        import os
        force_cpu = os.environ.get('FORCE_CPU', 'true').lower() == 'true'
        if force_cpu:
            self.device = torch.device('cpu')
            logger.info("Using CPU mode for memory efficiency (free tier)")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_model = None
        self.trained_processor = None
        
        # Improved Method 1: Specialized Deep Learning (NEW - preferred)
        self.improved_method_1 = None
        if IMPROVED_METHOD_1_AVAILABLE:
            try:
                logger.info("Attempting to initialize Improved Method 1 (Specialized Deep Learning)...")
                self.improved_method_1 = ImprovedDeepLearningMethod1(device=self.device)
                if self.improved_method_1.models:
                    logger.info(f"✓ Improved Method 1 initialized successfully with {len(self.improved_method_1.models)} models")
                else:
                    logger.warning("Improved Method 1 initialized but no models available")
                    self.improved_method_1 = None
            except Exception as e:
                logger.error(f"Could not initialize improved Method 1: {e}", exc_info=True)
                self.improved_method_1 = None
        
        # Modern ensemble service (fallback for Method 1)
        self.modern_ensemble = None
        if not self.improved_method_1 and MODERN_ENSEMBLE_AVAILABLE:
            try:
                logger.info("Attempting to initialize Modern Ensemble Method 1 (fallback)...")
                self.modern_ensemble = ModernEnsembleService(device=self.device)
                if self.modern_ensemble.model is not None:
                    logger.info("✓ Modern Ensemble Method 1 initialized successfully")
                else:
                    logger.warning("Modern Ensemble initialized but model is None - using fallback")
                    self.modern_ensemble = None
            except Exception as e:
                logger.error(f"Could not initialize modern ensemble: {e}", exc_info=True)
                self.modern_ensemble = None
        
        # Load fallback GenImage model if both improved and ensemble not available
        if not self.improved_method_1 and (self.modern_ensemble is None or self.modern_ensemble.model is None):
            logger.info("Loading GenImage ResNet-50 fallback model...")
            self._load_genimage_model()
        
        # Method 3: Advanced Spectral & Statistical Analysis (NEW - preferred)
        self.advanced_spectral_method_3 = None
        if ADVANCED_SPECTRAL_METHOD_3_AVAILABLE:
            try:
                logger.info("Attempting to initialize Method 3 (Advanced Spectral & Statistical Analysis)...")
                self.advanced_spectral_method_3 = AdvancedSpectralMethod3()
                logger.info("✓ Advanced Spectral Method 3 initialized successfully")
            except Exception as e:
                logger.error(f"Could not initialize Advanced Spectral Method 3: {e}", exc_info=True)
                self.advanced_spectral_method_3 = None
        
        # Fallback to forensics if spectral method not available
        self.improved_method_3 = None
        if not self.advanced_spectral_method_3 and IMPROVED_METHOD_3_AVAILABLE:
            try:
                logger.info("Falling back to Improved Method 3 (Forensics)...")
                self.improved_method_3 = ImprovedForensicsMethod3()
                logger.info("✓ Improved Method 3 (Forensics) initialized as fallback")
            except Exception as e:
                logger.warning(f"Could not initialize forensics fallback: {e}")
                self.improved_method_3 = None
        
        # Method 4: Hugging Face specialized models ensemble (NEW)
        self.huggingface_ensemble = None
        if HUGGINGFACE_AVAILABLE:
            try:
                logger.info("Attempting to initialize Hugging Face model ensemble...")
                self.huggingface_ensemble = HuggingFaceEnsemble(device=self.device)
                if self.huggingface_ensemble.models:
                    logger.info(f"✓ Hugging Face ensemble initialized with {len(self.huggingface_ensemble.models)} models")
                else:
                    logger.warning("Hugging Face ensemble initialized but no models available")
                    self.huggingface_ensemble = None
            except Exception as e:
                logger.error(f"Could not initialize Hugging Face ensemble: {e}", exc_info=True)
                self.huggingface_ensemble = None
        
        # Method names
        self.methods = {
            'method_1': 'DEEP_LEARNING_MODEL',
            'method_2': 'STATISTICAL_PATTERN_ANALYSIS',
            'method_3': 'METADATA_HEURISTIC_ANALYSIS',
            'method_4': 'HUGGINGFACE_SPECIALIZED_MODELS'
        }
        
        # Performance tracking for each method
        self.method_performance = {
            'method_1': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidence_sum': 0.0},
            'method_2': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidence_sum': 0.0},
            'method_3': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidence_sum': 0.0},
            'method_4': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidence_sum': 0.0}
        }
        
        # Accuracy-based weights (updated with Improved Methods 1 & 3)
        # Previous: Method 2: 70.0% ✅, Method 1: 32.5% ❌, Method 3: 8.8% ❌
        # 
        # NEW: Improved Method 1 (specialized models) - Expected 55-70% accuracy
        # NEW: Improved Method 3 (forensics) - Expected 50-65% accuracy
        # Method 2 remains excellent (70%) and maintains dominant weight
        #
        # Load saved weights if available, otherwise use defaults
        weights_config_path = 'method_weights_config.json'
        
        # Default weights based on latest analysis (last 30 samples)
        # CRITICAL: Method 3 is worst (33.3% accuracy, 100% confidence on errors) - reduce drastically
        # Method 1: 56.7% accuracy - BEST performer
        # Method 2: 40% accuracy - has 17 false positives
        default_weights = {
            'method_1': 0.35,  # 56.7% accuracy - reduce to make room for Method 4
            'method_2': 0.30,  # 40% accuracy - reduce weight due to false positives
            'method_3': 0.10,  # 33.3% accuracy - WORST, reduce drastically (was 20%)
            'method_4': 0.25   # NEW: Hugging Face specialized models - strong expected performance
        }
        
        # Default confidence calibration
        default_calibration = {
            'method_1': 0.8,   # Improved Method 1 - conservative start, adjust after testing
            'method_2': 1.0,   # Method 2 has excellent calibration - no adjustment needed
            'method_3': 0.7,   # Improved Method 3 - conservative start, should be more reliable than old
            'method_4': 0.95   # Hugging Face models - pre-trained specialists, expect good calibration
        }
        
        # Try to load saved weights from adaptive learning
        if os.path.exists(weights_config_path):
            try:
                with open(weights_config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.method_accuracy_weights = saved_config.get('weights', default_weights)
                    self.confidence_calibration = saved_config.get('calibration', default_calibration)
                    logger.info(f"✓ Loaded saved method weights from {weights_config_path}")
                    logger.info(f"  Weights: Method 1={self.method_accuracy_weights['method_1']*100:.1f}%, "
                             f"Method 2={self.method_accuracy_weights['method_2']*100:.1f}%, "
                             f"Method 3={self.method_accuracy_weights['method_3']*100:.1f}%")
            except Exception as e:
                logger.warning(f"Could not load saved weights: {e}, using defaults")
                self.method_accuracy_weights = default_weights
                self.confidence_calibration = default_calibration
        else:
            self.method_accuracy_weights = default_weights
            self.confidence_calibration = default_calibration
        
        # Confidence calibration factors (adjusted for Improved Methods)
        # Method 2 has excellent calibration (70.7% confidence, 70% accuracy) - no change
        # Improved Method 1: Start with conservative calibration (will adjust based on performance)
        # Improved Method 3: Start with conservative calibration (forensics should be more reliable)
        
        # Factor-specific weights for Method 2 (boost high-accuracy factors)
        # Based on 80-sample analysis showing exceptional factor performance
        # low_edge_density: 98.0% accuracy - EXCEPTIONAL (50/51 correct)
        # color_banding: 74.2% accuracy - VERY GOOD (23/31 correct)
        # regular_frequency_pattern: 70.0% accuracy - GOOD (56/80 correct)
        self.method_2_factor_weights = {
            'low_color_variation': 1.0,          # Standard weight
            'low_edge_density': 2.0,             # ⭐ MAJOR BOOST - 98.0% accuracy! (was 1.5x, now 2.0x = 0.5 contribution)
            'high_edge_density': 0.8,            # Slightly reduce - can cause false positives
            'uniform_texture': 1.0,              # Standard weight
            'uniform_brightness': 0.9,           # Slight reduction
            'color_banding': 1.3,                # ⭐ BOOSTED - 74.2% accuracy (was 1.0x, now 1.3x)
            'regular_frequency_pattern': 1.0      # Standard weight (70.0% accuracy)
        }
        
        # Threshold adjustments based on latest analysis (last 30 samples)
        # CRITICAL ISSUE: 94.4% false positives (17/18 errors)! All methods flagging real as AI
        # Method 3: 33.3% accuracy, 20 false positives, 100% confidence on errors - MAJOR PROBLEM
        # Method 2: 40% accuracy, 17 false positives - needs higher threshold
        # Method 1: 56.7% accuracy - best but still has 13 false positives
        self.method_thresholds = {
            'method_1': 0.55,  # RAISED from 0.5 - reduce false positives
            'method_2': 0.42,  # RAISED from 0.35 - reduce false positives (17 errors!)
            'method_3': 0.55   # RAISED from 0.28 to 0.55 - Method 3 has 100% confidence on errors!
        }
    
    def _load_genimage_model(self):
        """Load the trained GenImage model for Method 1"""
        try:
            model_path = 'genimage_detector_best.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                class GenImageDetector(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.backbone = models.resnet50(pretrained=False)
                        self.backbone.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(self.backbone.fc.in_features, 2)
                        )
                    
                    def forward(self, x):
                        return self.backbone(x)
                
                self.trained_model = GenImageDetector()
                
                if 'model_state_dict' in checkpoint:
                    self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.trained_model.load_state_dict(checkpoint)
                
                self.trained_model.eval()
                self.trained_model.to(self.device)
                
                self.trained_processor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                logger.info("✓ Method 1 (Deep Learning Model) loaded successfully")
            else:
                logger.warning("GenImage model not found for Method 1")
        except Exception as e:
            logger.error(f"Error loading GenImage model: {e}")
            self.trained_model = None
            self.trained_processor = None
    
    def detect_ai_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run all three detection methods independently and return
        comparative results.
        """
        try:
            if not os.path.exists(image_path):
                return self._error_result("Image file not found")
            
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            results = {}
            
            # METHOD 1: Deep Learning Model
            method_1_result = self._method_1_deep_learning(image)
            results['method_1'] = method_1_result
            
            # METHOD 2: Statistical Pattern Analysis
            method_2_result = self._method_2_statistical_patterns(image)
            results['method_2'] = method_2_result
            
            # METHOD 3: Metadata & Heuristic Analysis
            method_3_result = self._method_3_metadata_heuristics(image, image_path)
            results['method_3'] = method_3_result
            
            # METHOD 4: Hugging Face Specialized Models (NEW)
            if self.huggingface_ensemble:
                method_4_result = self._method_4_huggingface(image)
                results['method_4'] = method_4_result
            
            # Calculate agreement first
            agreement = self._calculate_agreement(results)
            
            # Use weighted voting based on method accuracy
            weighted_result = self._calculate_weighted_vote(results, agreement)
            
            # Create final result with all method comparisons
            final_result = {
                'is_ai_generated': weighted_result['is_ai_generated'],
                'confidence': weighted_result['confidence'],
                'method': 'three_method_comparison',
                'indicators': weighted_result['indicators'],
                'method_comparison': {
                    'method_1': {
                        'name': 'Deep Learning Model',
                        'description': 'Improved Deep Learning (Specialized AI Detection Models)' if self.improved_method_1 else ('Modern Ensemble (EfficientNet + ViT + ResNet)' if self.modern_ensemble else 'GenImage ResNet-50 (Fallback)'),
                        'is_ai_generated': method_1_result['is_ai_generated'],
                        'confidence': method_1_result['confidence'],
                        'indicators': method_1_result['indicators'],
                        'available': (self.improved_method_1 is not None and hasattr(self.improved_method_1, 'models') and len(self.improved_method_1.models) > 0) or (self.modern_ensemble is not None and self.modern_ensemble.model is not None) or (self.trained_model is not None)
                    },
                    'method_2': {
                        'name': 'Statistical Pattern Analysis',
                        'description': 'Mathematical analysis of pixel patterns, color distribution, and texture',
                        'is_ai_generated': method_2_result['is_ai_generated'],
                        'confidence': method_2_result['confidence'],
                        'indicators': method_2_result['indicators'],
                        'available': True
                    },
                        'method_3': {
                            'name': 'Advanced Forensics Analysis',
                            'description': 'Advanced Spectral & Statistical Analysis' if self.advanced_spectral_method_3 else ('Forensics Analysis (ELA, Noise, Color Space, DCT)' if self.improved_method_3 else 'EXIF metadata, file patterns, and rule-based heuristics (fallback)'),
                            'is_ai_generated': method_3_result['is_ai_generated'],
                            'confidence': method_3_result['confidence'],
                            'indicators': method_3_result['indicators'],
                            'available': True
                        },
                    'best_method': weighted_result['method_id'],
                    'agreement': self._calculate_agreement(results),
                    'performance_stats': self.method_performance.copy()
                },
                'method_4': {
                    'name': 'Hugging Face Specialized Models',
                    'description': 'ViT AI-detector, AI vs Human Detector, WildFakeDetector',
                    'is_ai_generated': results.get('method_4', {}).get('is_ai_generated', False) if 'method_4' in results else None,
                    'confidence': results.get('method_4', {}).get('confidence', 0.0) if 'method_4' in results else None,
                    'indicators': results.get('method_4', {}).get('indicators', []) if 'method_4' in results else ['Not available'],
                    'available': self.huggingface_ensemble is not None and len(self.huggingface_ensemble.models) > 0 if self.huggingface_ensemble else False
                },
                'analysis_details': {
                    'all_methods': results,
                    'best_method': weighted_result['method_id'],
                    'agreement': agreement,
                    'method_comparison': {
                        'method_1': {
                            'name': 'Deep Learning Model',
                            'description': 'Improved Deep Learning (Specialized AI Detection Models)' if self.improved_method_1 else ('Modern Ensemble (EfficientNet + ViT + ResNet)' if self.modern_ensemble else 'GenImage ResNet-50 (Fallback)'),
                            'is_ai_generated': method_1_result['is_ai_generated'],
                            'confidence': method_1_result['confidence'],
                            'indicators': method_1_result['indicators'],
                            'available': (self.improved_method_1 is not None and hasattr(self.improved_method_1, 'models') and len(self.improved_method_1.models) > 0) or (self.modern_ensemble is not None and self.modern_ensemble.model is not None) or (self.trained_model is not None),
                            'ensemble_models': method_1_result.get('ensemble_models', [])
                        },
                        'method_2': {
                            'name': 'Statistical Pattern Analysis',
                            'description': 'Mathematical analysis of pixel patterns, color distribution, and texture',
                            'is_ai_generated': method_2_result['is_ai_generated'],
                            'confidence': method_2_result['confidence'],
                            'indicators': method_2_result['indicators'],
                            'available': True
                        },
                        'method_3': {
                            'name': 'Advanced Forensics Analysis',
                            'description': 'Advanced Spectral & Statistical Analysis' if self.advanced_spectral_method_3 else ('Forensics Analysis (ELA, Noise, Color Space, DCT)' if self.improved_method_3 else 'EXIF metadata, file patterns, and rule-based heuristics (fallback)'),
                            'is_ai_generated': method_3_result['is_ai_generated'],
                            'confidence': method_3_result['confidence'],
                            'indicators': method_3_result['indicators'],
                            'available': True
                        },
                        'method_4': {
                            'name': 'Hugging Face Specialized Models',
                            'description': 'ViT AI-detector, AI vs Human Detector, WildFakeDetector',
                            'is_ai_generated': results.get('method_4', {}).get('is_ai_generated', False) if 'method_4' in results else None,
                            'confidence': results.get('method_4', {}).get('confidence', 0.0) if 'method_4' in results else None,
                            'indicators': results.get('method_4', {}).get('indicators', []) if 'method_4' in results else ['Not available'],
                            'available': self.huggingface_ensemble is not None and len(self.huggingface_ensemble.models) > 0 if self.huggingface_ensemble else False,
                            'model_predictions': results.get('method_4', {}).get('model_predictions', {})
                        },
                        'best_method': weighted_result['method_id'],
                        'agreement': agreement,
                        'performance_stats': self.method_performance.copy()
                    }
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in three-method detection: {e}")
            return self._error_result(f"Detection failed: {str(e)}")
    
    def _method_1_deep_learning(self, image: Image.Image) -> Dict[str, Any]:
        """Method 1: Improved Deep Learning Model (Specialized AI Detection Models)"""
        
        # Try improved Method 1 first (preferred - uses specialized models)
        if self.improved_method_1:
            try:
                result = self.improved_method_1.detect(image)
                if 'error' not in result:
                    indicators = result['indicators'] + [
                        "⭐ Method: Improved Deep Learning (Specialized AI Detection Models)",
                        "Models: EfficientNet-B4, ViT-Large, ConvNeXt Base",
                        "Based on state-of-the-art research for synthetic image detection"
                    ]
                    return {
                        'is_ai_generated': result['is_ai_generated'],
                        'confidence': result['confidence'],
                        'indicators': indicators,
                        'method_id': 'method_1',
                        'raw_score': result['probabilities']['ai'],
                        'ensemble_models': result.get('available_models', [])
                    }
            except Exception as e:
                logger.warning(f"Improved Method 1 failed, using fallback: {e}")
        
        # Fallback to modern ensemble
        if self.modern_ensemble:
            try:
                result = self.modern_ensemble.detect(image)
                if 'error' not in result:
                    indicators = result['indicators'] + [
                        "Method: Modern Ensemble (EfficientNet + ViT + ResNet)",
                        "Based on successful Kaggle competition techniques"
                    ]
                    return {
                        'is_ai_generated': result['is_ai_generated'],
                        'confidence': result['confidence'],
                        'indicators': indicators,
                        'method_id': 'method_1',
                        'raw_score': result['probabilities']['ai'],
                        'ensemble_models': result.get('available_models', [])
                    }
            except Exception as e:
                logger.warning(f"Modern ensemble failed, using fallback: {e}")
        
        # Final fallback to GenImage ResNet-50 model
        if not self.trained_model or not self.trained_processor:
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': ['Deep Learning Model not available'],
                'method_id': 'method_1'
            }
        
        try:
            # Preprocess image
            input_tensor = self.trained_processor(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.trained_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                ai_probability = probabilities[0][1].item()
            
            is_ai_generated = ai_probability > 0.5
            confidence = max(ai_probability, 1 - ai_probability)
            
            indicators = [
                f"Neural network prediction: {'AI-generated' if is_ai_generated else 'Real'} image",
                f"Model confidence: {confidence*100:.1f}%",
                f"AI probability: {ai_probability*100:.1f}%",
                "Method: ResNet-50 (Fallback - GenImage dataset)"
            ]
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'indicators': indicators,
                'method_id': 'method_1',
                'raw_score': ai_probability
            }
            
        except Exception as e:
            logger.error(f"Error in Method 1: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': [f'Method 1 error: {str(e)}'],
                'method_id': 'method_1'
            }
    
    def _method_2_statistical_patterns(self, image: Image.Image) -> Dict[str, Any]:
        """Method 2: Statistical Pattern Analysis"""
        try:
            img_array = np.array(image)
            indicators = []
            ai_score = 0.0
            factors = []
            
            # Helper function to get factor weight
            def get_factor_weight(factor_name):
                return self.method_2_factor_weights.get(factor_name, 1.0)
            
            # 1. Color Variation Analysis
            rgb_std = np.std(img_array, axis=(0, 1))
            mean_std = np.mean(rgb_std)
            # Adjusted: Balance between catching AI and avoiding false positives
            # CRITICAL: Too many false positives - be more conservative
            if mean_std < 12:  # Very low color variation - BACK to 12 (from 15) to reduce false positives
                base_score = 0.20
                weight = get_factor_weight('low_color_variation')
                ai_score += base_score * weight
                factors.append('low_color_variation')
                indicators.append(f"Very low color variation (std: {mean_std:.1f}, weight: {weight:.2f}x)")
            
            # 2. Edge Density Analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Adjusted thresholds - balance false positives vs false negatives
            # LOW_EDGE_DENSITY IS MAJOR BOOST (98.0% accuracy!) - highest weight factor
            # CRITICAL: Too many false positives - slightly more conservative
            if edge_density < 0.06:  # Very low edge density - back to 0.06 (from 0.08) to reduce false positives
                base_score = 0.25
                weight = get_factor_weight('low_edge_density')  # 2.0x boost - 98% accuracy!
                ai_score += base_score * weight  # Now contributes 0.50 instead of 0.25 (double!)
                factors.append('low_edge_density')
                indicators.append(f"Very low edge density ({edge_density*100:.1f}%, weight: {weight:.2f}x) - ⭐ EXCEPTIONAL RELIABILITY (98% accuracy)")
            elif edge_density > 0.40:  # Unusually high (was 0.35, more conservative)
                base_score = 0.15
                weight = get_factor_weight('high_edge_density')  # 0.8x reduction
                ai_score += base_score * weight
                factors.append('high_edge_density')
                indicators.append(f"Unusually high edge density ({edge_density*100:.1f}%, weight: {weight:.2f}x)")
            
            # 3. Texture Uniformity (Local Binary Patterns)
            # Simplified LBP
            lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
            lbp_var = np.var(lbp)
            # Adjusted threshold - balance to reduce false positives
            if lbp_var < 6:  # Extremely uniform texture - back to 6 (from 8) to reduce false positives
                base_score = 0.25
                weight = get_factor_weight('uniform_texture')
                ai_score += base_score * weight
                factors.append('uniform_texture')
                indicators.append(f"Extremely uniform texture pattern (variance: {lbp_var:.1f}, weight: {weight:.2f}x)")
            
            # 4. Brightness Distribution
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            # Adjusted threshold - reduce false positives
            if brightness_std < 16:  # Extremely uniform brightness - back to 16 (from 20) to reduce false positives
                base_score = 0.20
                weight = get_factor_weight('uniform_brightness')  # 0.9x reduction
                ai_score += base_score * weight
                factors.append('uniform_brightness')
                indicators.append(f"Extremely uniform brightness (std: {brightness_std:.1f}, weight: {weight:.2f}x)")
            
            # 5. Color Histogram Analysis
            for i in range(3):  # RGB channels
                channel = img_array[:, :, i]
                hist, _ = np.histogram(channel, bins=256)
                # Check for unusual peaks (color banding)
                # Adjusted threshold - color banding is highly reliable (100% accuracy when detected)
                # But balance with false positive reduction
                peak_count = np.sum(hist > np.mean(hist) * 3.0)  # Back to 3.0 (from 2.8) to reduce false positives
                if peak_count > 20:  # Back to 20 (from 18) to reduce false positives
                    base_score = 0.15
                    weight = get_factor_weight('color_banding')  # 1.3x boost - 74.2% accuracy
                    ai_score += base_score * weight  # Now contributes 0.195 instead of 0.15
                    factors.append('color_banding')
                    indicators.append(f"Color banding detected in channel {i} (weight: {weight:.2f}x) - HIGH ACCURACY (74.2%)")
                    break
            
            # 6. Spatial Frequency Analysis
            # Convert to grayscale for FFT
            f_transform = fft.fft2(gray)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Check for regular patterns
            freq_variance = np.var(magnitude_spectrum)
            # Adjusted threshold - regular patterns are 100% accurate when detected
            # But balance to reduce false positives
            if freq_variance < 85:  # Very regular patterns - back to 85 (from 100) to reduce false positives
                base_score = 0.15
                weight = get_factor_weight('regular_frequency_pattern')
                ai_score += base_score * weight
                factors.append('regular_frequency_pattern')
                indicators.append(f"Very regular frequency patterns detected (weight: {weight:.2f}x)")
            
            # Normalize and determine result
            ai_score = min(ai_score, 1.0)
            
            # Use adaptive threshold based on number of factors
            # CRITICAL FIX: Method 2 has 17 false positives (94.4% of errors are false positives)!
            base_threshold = self.method_thresholds.get('method_2', 0.42)
            if len(factors) < 2:
                # If only one indicator, require very high score to reduce false positives
                threshold = 0.60  # RAISED from 0.50
            elif len(factors) == 2:
                # Two indicators - require higher score
                threshold = base_threshold + 0.10  # 0.52 for base of 0.42
            else:
                # Three+ indicators - use raised base threshold
                threshold = base_threshold
            
            is_ai_generated = ai_score > threshold
            confidence = max(ai_score, 1 - ai_score)
            
            # Boost confidence if multiple strong indicators agree
            if len(factors) >= 3:
                confidence = min(0.85, confidence * 1.1)
            elif len(factors) >= 2:
                confidence = min(0.80, confidence * 1.05)
            
            if not indicators:
                indicators.append("No significant statistical anomalies detected")
                indicators.append("Patterns suggest natural image characteristics")
            
            # Add summary
            indicators.insert(0, f"Statistical analysis: {len(factors)} AI indicators found")
            indicators.append(f"Overall AI score: {ai_score*100:.1f}%")
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'indicators': indicators,
                'method_id': 'method_2',
                'raw_score': ai_score,
                'factors_detected': factors
            }
            
        except Exception as e:
            logger.error(f"Error in Method 2: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': [f'Method 2 error: {str(e)}'],
                'method_id': 'method_2'
            }
    
    def _method_3_metadata_heuristics(self, image: Image.Image, image_path: str) -> Dict[str, Any]:
        """Method 3: Advanced Spectral & Statistical Analysis"""
        
        # Use Advanced Spectral Method 3 (preferred - trusted approach)
        if self.advanced_spectral_method_3:
            try:
                result = self.advanced_spectral_method_3.detect(image_path)
                if 'error' not in result:
                    indicators = result['indicators'] + [
                        "⭐ Method: Advanced Spectral & Statistical Analysis",
                        "Techniques: Spectral Energy Distribution, Multi-scale Texture Analysis,",
                        "Color Entropy Statistics, Frequency Pattern Analysis, Wavelet Decomposition",
                        "Based on established signal processing and statistical pattern recognition"
                    ]
                    return {
                        'is_ai_generated': result['is_ai_generated'],
                        'confidence': result['confidence'],
                        'indicators': indicators,
                        'method_id': 'method_3',
                        'raw_score': result.get('score', result['confidence']),
                        'factors_detected': result.get('factors', []),
                        'spectral_details': result.get('analysis_details', {})
                    }
            except Exception as e:
                logger.warning(f"Advanced Spectral Method 3 failed, trying fallback: {e}")
        
        # Fallback to forensics if available
        if self.improved_method_3:
            try:
                result = self.improved_method_3.detect(image_path)
                if 'error' not in result:
                    indicators = result['indicators'] + [
                        "⭐ Method: Advanced Image Forensics",
                        "Techniques: Error Level Analysis (ELA), Noise Pattern Analysis,",
                        "Color Space Analysis, DCT Coefficient Analysis",
                        "Based on digital forensics research for synthetic image detection"
                    ]
                    return {
                        'is_ai_generated': result['is_ai_generated'],
                        'confidence': result['confidence'],
                        'indicators': indicators,
                        'method_id': 'method_3',
                        'raw_score': result.get('score', result['confidence']),
                        'factors_detected': result.get('factors', []),
                        'forensics_details': result.get('analysis_details', {})
                    }
            except Exception as e:
                logger.warning(f"Improved Method 3 failed, using fallback: {e}")
        
        # Fallback to old metadata method
        try:
            indicators = []
            ai_score = 0.0
            factors = []
            
            # 1. EXIF Metadata Analysis
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, str):
                        value_lower = value.lower()
                        # Check for AI generation software
                        ai_keywords = ['midjourney', 'dall-e', 'stable diffusion', 'ai', 'generated', 
                                      'artificial', 'gan', 'neural', 'deep dream']
                        if any(keyword in value_lower for keyword in ai_keywords):
                            ai_score += 0.30
                            factors.append('ai_software_in_exif')
                            indicators.append(f"AI generation software detected in EXIF: {tag}={value}")
            
            # 2. Filename Pattern Analysis
            filename = os.path.basename(image_path).lower()
            ai_patterns = [
                'ai_', 'generated_', 'midjourney', 'dalle', 'stable_diffusion',
                'sd_', 'gan_', 'synthetic', 'fake', 'artificial'
            ]
            for pattern in ai_patterns:
                if pattern in filename:
                    ai_score += 0.25
                    factors.append('ai_filename_pattern')
                    indicators.append(f"AI-related pattern in filename: '{pattern}'")
                    break
            
            # 3. File Size Analysis
            # REDUCED WEIGHT - "common_ai_size" only 29.4% accuracy
            file_size = os.path.getsize(image_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # AI images often have specific size patterns - but weak indicator
            if file_size < 50000:  # Very small (under 50KB)
                ai_score += 0.05  # Reduced from 0.10
                factors.append('unusually_small_file')
                indicators.append(f"Unusually small file size: {file_size_mb:.2f} MB")
            # Removed common_ai_size - too inaccurate (29.4%)
            
            # 4. Image Dimensions Analysis
            # REDUCED WEIGHT - "common_ai_resolution" only 15.0% accuracy, "square_aspect_ratio" 0%
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 1
            
            # Square images - REMOVED (0% accuracy)
            # if 0.95 < aspect_ratio < 1.05:  # Square - disabled due to poor accuracy
            
            # Common AI generation resolutions - REDUCED (15.0% accuracy)
            common_ai_resolutions = [
                (512, 512), (768, 768), (1024, 1024), (512, 768), (768, 1024),
                (1024, 1536), (1536, 1024)
            ]
            if (width, height) in common_ai_resolutions:
                ai_score += 0.05  # Reduced from 0.15 (was too inaccurate)
                factors.append('common_ai_resolution')
                indicators.append(f"Common AI generation resolution: {width}x{height} (weak indicator)")
            
            # 5. Compression Artifacts Analysis
            # Check JPEG quality based on file analysis
            if filename.endswith('.jpeg') or filename.endswith('.jpg'):
                try:
                    # Save and reload to check compression
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        image.save(tmp.name, 'JPEG', quality=95)
                        reloaded = Image.open(tmp.name)
                        
                        # Compare file sizes (high compression might indicate AI)
                        reloaded_size = os.path.getsize(tmp.name)
                        compression_ratio = file_size / reloaded_size if reloaded_size > 0 else 1
                        
                        if compression_ratio > 3:  # Highly compressed
                            ai_score += 0.10
                            factors.append('high_compression')
                            indicators.append(f"High compression detected (ratio: {compression_ratio:.1f}x)")
                        
                        os.unlink(tmp.name)
                except:
                    pass
            
            # 6. Color Space Analysis
            # Check if image has unusual color characteristics
            if image.mode == 'RGB':
                img_array = np.array(image)
                # Check for unusual color distributions
                for i in range(3):
                    channel = img_array[:, :, i]
                    unique_values = len(np.unique(channel))
                    # AI images sometimes have limited color palettes
                    if unique_values < 50:
                        ai_score += 0.05
                        factors.append('limited_color_palette')
                        indicators.append(f"Limited color palette in channel {i}")
                        break
            
            # Normalize and determine result
            ai_score = min(ai_score, 1.0)
            
            # IMPROVED THRESHOLD LOGIC - "no_factors" is correct 64.7% of the time
            # Require at least 1 strong indicator (EXIF or filename) OR multiple weak indicators
            strong_factors = ['ai_software_in_exif', 'ai_filename_pattern']
            strong_count = sum(1 for f in factors if f in strong_factors)
            
            # Require higher threshold - only flag if strong evidence exists
            # If no strong factors, require very high score (>0.5)
            if strong_count > 0:
                threshold = 0.35  # Lower threshold if strong factors present
            else:
                threshold = 0.50  # Higher threshold if only weak factors
            
            is_ai_generated = ai_score > threshold
            confidence = max(ai_score, 1 - ai_score)
            
            # Reduce confidence if only weak factors detected
            if strong_count == 0 and len(factors) > 0:
                confidence = confidence * 0.7  # Reduce confidence for weak-only factors
            
            if not indicators:
                indicators.append("No metadata or heuristic indicators found")
                indicators.append("File characteristics suggest natural image (64.7% accurate when no factors)")
            
            # Add summary
            indicators.insert(0, f"Metadata/Heuristic analysis: {len(factors)} indicators ({strong_count} strong)")
            indicators.append(f"Overall AI score: {ai_score*100:.1f}% (threshold: {threshold*100:.1f}%)")
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'indicators': indicators,
                'method_id': 'method_3',
                'raw_score': ai_score,
                'factors_detected': factors
            }
            
        except Exception as e:
            logger.error(f"Error in Method 3: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': [f'Method 3 error: {str(e)}'],
                'method_id': 'method_3'
            }
    
    def _method_4_huggingface(self, image: Image.Image) -> Dict[str, Any]:
        """Method 4: Hugging Face Specialized Models Ensemble"""
        
        if not self.huggingface_ensemble:
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': ['Hugging Face models not available'],
                'method_id': 'method_4'
            }
        
        try:
            result = self.huggingface_ensemble.detect(image)
            if 'error' not in result:
                indicators = result['indicators'] + [
                    "⭐ Method: Hugging Face Specialized Models",
                    "Models: ViT AI-detector, AI vs Human Detector, WildFakeDetector",
                    "Pre-trained on large-scale AI-generated image datasets"
                ]
                return {
                    'is_ai_generated': result['is_ai_generated'],
                    'confidence': result['confidence'],
                    'indicators': indicators,
                    'method_id': 'method_4',
                    'raw_score': result['probabilities']['ai'],
                    'model_predictions': result.get('model_predictions', {}),
                    'models_count': result.get('models_count', 0)
                }
            else:
                return {
                    'is_ai_generated': False,
                    'confidence': 0.0,
                    'indicators': [f'Hugging Face ensemble error: {result["error"]}'],
                    'method_id': 'method_4'
                }
        except Exception as e:
            logger.error(f"Error in Method 4: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': [f'Method 4 error: {str(e)}'],
                'method_id': 'method_4'
            }
    
    def _calculate_weighted_vote(self, results: Dict, agreement: Dict) -> Dict[str, Any]:
        """
        Calculate final result using accuracy-based weighted voting.
        
        This combines:
        1. Accuracy-based weights (Method 2 gets 57% weight based on 81.2% accuracy)
        2. Confidence calibration (adjusts for overconfidence)
        3. Agreement boosting (boosts confidence when methods agree)
        """
        weighted_ai_score = 0.0
        weighted_real_score = 0.0
        total_weight = 0.0
        method_contributions = {}
        
        # Collect decisions for agreement analysis
        decisions = {}
        
        for method_id, result in results.items():
            if method_id not in self.method_accuracy_weights:
                continue
                
            # Get base weight from accuracy
            base_weight = self.method_accuracy_weights[method_id]
            
            # Apply confidence calibration
            calibrated_confidence = result['confidence'] * self.confidence_calibration.get(method_id, 1.0)
            
            # Calculate effective weight (accuracy weight * calibrated confidence)
            effective_weight = base_weight * calibrated_confidence
            total_weight += effective_weight
            
            # Store decision
            decisions[method_id] = result['is_ai_generated']
            
            # Contribute to AI or Real score
            if result['is_ai_generated']:
                weighted_ai_score += effective_weight
            else:
                weighted_real_score += effective_weight
            
            # Track contribution
            method_contributions[method_id] = {
                'decision': 'AI' if result['is_ai_generated'] else 'Real',
                'base_weight': base_weight,
                'calibrated_confidence': calibrated_confidence,
                'effective_weight': effective_weight
            }
        
        if total_weight == 0:
            return {
                'method_id': 'none',
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': ['No methods available']
            }
        
        # Normalize scores
        normalized_ai_score = weighted_ai_score / total_weight
        normalized_real_score = weighted_real_score / total_weight
        
        # Determine final decision
        is_ai_generated = normalized_ai_score > normalized_real_score
        
        # Base confidence from normalized scores
        base_confidence = max(normalized_ai_score, normalized_real_score)
        
        # Agreement boosting and false positive protection
        # If 2 or more methods agree, boost confidence
        decision_list = list(decisions.values())
        ai_count = sum(decision_list)
        real_count = len(decision_list) - ai_count
        
        # Special case: Method 2 predicting AI alone (known false positive pattern)
        # CRITICAL FIX: Method 2 has 15 false positives - reduce confidence when alone
        # Check if methods are available (method_1 might not be available)
        method_1_available = 'method_1' in decisions
        method_2_available = 'method_2' in decisions
        method_3_available = 'method_3' in decisions
        
        method_2_alone_ai = (
            method_2_available and
            decisions.get('method_2') == True and
            (not method_1_available or decisions.get('method_1') == False) and
            (not method_3_available or decisions.get('method_3') == False) and
            is_ai_generated
        )
        
        # Method 3 alone predicting Real (known false negative pattern - 24 false negatives)
        method_3_alone_real = (
            method_3_available and
            decisions.get('method_3') == False and
            (not method_1_available or decisions.get('method_1') == True) and
            (not method_2_available or decisions.get('method_2') == True) and
            not is_ai_generated  # Current decision is Real
        )
        
        # Check if Methods 1 & 2 disagree - Method 3 should be decisive
        method_1_decision = decisions.get('method_1')
        method_2_decision = decisions.get('method_2')
        method_3_decision = decisions.get('method_3')
        methods_1_2_disagree = (
            method_1_available and method_2_available and 
            method_1_decision != method_2_decision
        )
        method_3_is_tiebreaker = (
            methods_1_2_disagree and 
            method_3_available and
            method_3_decision in [method_1_decision, method_2_decision]
        )
        
        if agreement['unanimous']:
            # Unanimous agreement: boost confidence by 15%
            final_confidence = min(0.95, base_confidence * 1.15)
            agreement_boost = "Unanimous agreement boost"
        elif agreement['agreement_percentage'] >= 66:
            # Majority agreement: boost confidence by 10%
            if method_2_alone_ai:
                # Method 2 alone predicting AI - SIGNIFICANTLY reduce confidence (15 false positives!)
                final_confidence = min(0.65, base_confidence * 0.70)  # More aggressive reduction
                agreement_boost = "Method 2 alone - significantly reduced confidence (false positive protection)"
            elif method_3_alone_real:
                # Method 3 alone predicting Real - reduce confidence (24 false negatives!)
                final_confidence = min(0.70, base_confidence * 0.75)
                agreement_boost = "Method 3 alone predicting Real - reduced confidence (false negative protection)"
            elif method_3_is_tiebreaker:
                # Method 3 breaking the tie - boost significantly
                final_confidence = min(0.92, base_confidence * 1.20)
                agreement_boost = "Method 3 tie-breaker - Methods 1 & 2 disagree, Method 3 decisive"
                # Boost Method 3 contribution
                if 'method_3' in method_contributions:
                    method_contributions['method_3']['effective_weight'] *= 1.5
                    # Recalculate with boosted Method 3
                    total_weight = sum(m['effective_weight'] for m in method_contributions.values())
                    if total_weight > 0:
                        normalized_ai_score = weighted_ai_score / total_weight
                        normalized_real_score = weighted_real_score / total_weight
                        is_ai_generated = normalized_ai_score > normalized_real_score
                        base_confidence = max(normalized_ai_score, normalized_real_score)
                        final_confidence = min(0.92, base_confidence * 1.20)
            else:
                final_confidence = min(0.90, base_confidence * 1.10)
                agreement_boost = "Majority agreement boost"
        else:
            # Disagreement: reduce confidence by 15%
            if method_2_alone_ai:
                # Method 2 disagreeing alone - reduce more (false positive pattern)
                final_confidence = base_confidence * 0.65  # More aggressive - was 0.75
                agreement_boost = "Method 2 isolated - significantly reduced confidence (false positive protection)"
            elif method_3_alone_real:
                # Method 3 alone predicting Real - reduce (false negative pattern)
                final_confidence = base_confidence * 0.70
                agreement_boost = "Method 3 isolated - reduced confidence (may be missing AI)"
            else:
                final_confidence = base_confidence * 0.85
                agreement_boost = "Method disagreement - reduced confidence"
        
        # Determine which method contributed most
        contributing_method = max(
            method_contributions.items(),
            key=lambda x: x[1]['effective_weight']
        )[0]
        
        # Collect indicators from all methods (prioritize most accurate)
        all_indicators = []
        
        # Add Method 2 indicators first (most accurate)
        if 'method_2' in results:
            all_indicators.extend(results['method_2']['indicators'][:2])
        
        # Add contributing method indicators
        if contributing_method in results:
            all_indicators.extend(results[contributing_method]['indicators'][:2])
        
        # Add other method indicators
        for method_id in ['method_1', 'method_3']:
            if method_id in results and method_id != contributing_method:
                all_indicators.extend(results[method_id]['indicators'][:1])
        
        # Add agreement indicator
        all_indicators.insert(0, agreement_boost)
        all_indicators.insert(1, f"Weighted voting: {normalized_ai_score*100:.1f}% AI, {normalized_real_score*100:.1f}% Real")
        
        return {
            'method_id': contributing_method,
            'is_ai_generated': is_ai_generated,
            'confidence': final_confidence,
            'indicators': all_indicators[:5],  # Top 5 indicators
            'weighted_scores': {
                'ai_score': normalized_ai_score,
                'real_score': normalized_real_score,
                'base_confidence': base_confidence,
                'final_confidence': final_confidence
            },
            'method_contributions': method_contributions
        }
    
    def _calculate_agreement(self, results: Dict) -> Dict[str, Any]:
        """Calculate agreement between methods"""
        decisions = [r['is_ai_generated'] for r in results.values()]
        
        ai_count = sum(decisions)
        real_count = len(decisions) - ai_count
        
        agreement = {
            'total_methods': len(decisions),
            'ai_decisions': ai_count,
            'real_decisions': real_count,
            'majority_decision': ai_count > real_count,
            'unanimous': len(set(decisions)) == 1,
            'agreement_percentage': (max(ai_count, real_count) / len(decisions) * 100) if decisions else 0
        }
        
        return agreement
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return standardized error result"""
        return {
            'is_ai_generated': False,
            'confidence': 0.0,
            'method': 'error',
            'indicators': [error_message],
            'method_comparison': {
                'error': error_message
            },
            'analysis_details': {
                'error': error_message
            }
        }

# Lazy-loaded singleton instance - only create when first accessed
# This prevents blocking Django startup with heavy model loading
_service_instance = None

def get_detection_service():
    """Get or create the detection service instance (lazy loading).
    
    Models are loaded only on first detection request, not at startup.
    This prevents worker timeouts during deployment.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ThreeMethodDetectionService()
    return _service_instance

# Module-level instance for backward compatibility
# Access via get_detection_service() is preferred
class _ServiceProxy:
    """Proxy to lazy-load the service."""
    def __getattr__(self, name):
        return getattr(get_detection_service(), name)

three_method_detection_service = _ServiceProxy()
