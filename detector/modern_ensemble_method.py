"""
Modern Ensemble AI Detection Method
Based on successful Kaggle competition techniques using EfficientNet, ViT, and ResNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import logging
import os
import timm

logger = logging.getLogger(__name__)

class ModernEnsembleDetector(nn.Module):
    """
    Ensemble model combining EfficientNet, ViT, and ResNet
    Based on successful Kaggle competition approaches
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(ModernEnsembleDetector, self).__init__()
        
        # EfficientNet-B0: Good at detecting artifacts
        try:
            self.efficientnet = timm.create_model(
                'efficientnet_b0', 
                pretrained=True, 
                num_classes=num_classes,
                drop_rate=dropout
            )
            self.efficientnet_available = True
        except Exception as e:
            logger.warning(f"Could not load EfficientNet: {e}")
            self.efficientnet_available = False
        
        # Vision Transformer: Captures global patterns
        try:
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=num_classes,
                drop_rate=dropout
            )
            self.vit_available = True
        except Exception as e:
            logger.warning(f"Could not load ViT: {e}")
            self.vit_available = False
        
        # ResNet-50: Baseline model
        try:
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.resnet.fc.in_features, num_classes)
            )
            self.resnet_available = True
        except Exception as e:
            logger.warning(f"Could not load ResNet: {e}")
            self.resnet_available = False
        
        # Ensemble weights (can be adjusted based on validation performance)
        # Default weights: balance between models
        self.ensemble_weights = {
            'efficientnet': 0.40,  # Strong for artifact detection
            'vit': 0.40,            # Strong for global patterns
            'resnet': 0.20          # Baseline support
        }
    
    def forward(self, x):
        """
        Forward pass through ensemble
        Returns weighted average of all available models
        """
        predictions = []
        weights = []
        
        # Get predictions from each available model
        if self.efficientnet_available:
            eff_pred = F.softmax(self.efficientnet(x), dim=1)
            predictions.append(eff_pred)
            weights.append(self.ensemble_weights['efficientnet'])
        
        if self.vit_available:
            vit_pred = F.softmax(self.vit(x), dim=1)
            predictions.append(vit_pred)
            weights.append(self.ensemble_weights['vit'])
        
        if self.resnet_available:
            res_pred = F.softmax(self.resnet(x), dim=1)
            predictions.append(res_pred)
            weights.append(self.ensemble_weights['resnet'])
        
        if not predictions:
            raise RuntimeError("No models available in ensemble")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted ensemble
        ensemble_output = sum(w * pred for w, pred in zip(weights, predictions))
        
        return torch.log(ensemble_output + 1e-8)  # Return log probabilities


class ModernEnsembleService:
    """
    Service wrapper for modern ensemble detection
    """
    
    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize ensemble model
        try:
            logger.info("Initializing Modern Ensemble Detector...")
            self.model = ModernEnsembleDetector(num_classes=2, dropout=0.3)
            
            # Check which models are available
            available_count = sum([
                self.model.efficientnet_available,
                self.model.vit_available,
                self.model.resnet_available
            ])
            logger.info(f"Ensemble initialized with {available_count}/3 models available")
            
            # Load trained weights if available
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded ensemble model from {model_path}")
            else:
                logger.info("Using pretrained ensemble (no fine-tuning)")
            
            self.model.eval()
            self.model.to(self.device)
            logger.info("✓ Modern Ensemble Method initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Modern Ensemble: {e}", exc_info=True)
            self.model = None
    
    def detect(self, image: Image.Image) -> dict:
        """
        Detect if image is AI-generated using ensemble
        
        Returns:
            dict with 'is_ai_generated', 'confidence', 'probabilities', 'indicators'
        """
        if self.model is None:
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'error': 'Ensemble model not available',
                'indicators': ['Ensemble model failed to initialize']
            }
        
        try:
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get ensemble prediction
            with torch.no_grad():
                log_probs = self.model(input_tensor)
                probs = torch.exp(log_probs).cpu().numpy()[0]
            
            # Interpret results: class 0 = Real, class 1 = AI
            real_prob = float(probs[0])
            ai_prob = float(probs[1])
            
            # Determine prediction
            is_ai_generated = ai_prob > 0.5
            confidence = max(ai_prob, real_prob)
            
            # Create indicators
            indicators = [
                f"Ensemble prediction: {'AI-generated' if is_ai_generated else 'Real'}",
                f"AI probability: {ai_prob*100:.1f}%",
                f"Real probability: {real_prob*100:.1f}%",
            ]
            
            # Add model availability info
            available_models = []
            if self.model.efficientnet_available:
                available_models.append("EfficientNet-B0")
            if self.model.vit_available:
                available_models.append("ViT-Base")
            if self.model.resnet_available:
                available_models.append("ResNet-50")
            
            indicators.append(f"Active models: {', '.join(available_models)}")
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'probabilities': {
                    'real': real_prob,
                    'ai': ai_prob
                },
                'indicators': indicators,
                'available_models': available_models
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'error': str(e),
                'indicators': [f'Ensemble detection error: {str(e)}']
            }
    
    def update_ensemble_weights(self, weights: dict):
        """
        Update ensemble model weights based on validation performance
        
        Args:
            weights: Dict with 'efficientnet', 'vit', 'resnet' weights (should sum to 1.0)
        """
        if self.model:
            self.model.ensemble_weights.update(weights)
            logger.info(f"Updated ensemble weights: {weights}")

