import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageStat
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class AIImageDetectionService:
    def __init__(self):
        """Initialize the AI model for detecting AI-generated images"""
        try:
            # Use BLIP model for image analysis
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"AI Detection model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.processor = None
    
    def detect_ai_image(self, image_path):
        """
        Detect if an image is AI-generated or real
        """
        try:
            # Load and analyze image
            image = Image.open(image_path).convert('RGB')
            
            # Perform multiple analysis methods
            analysis_results = {
                'is_ai_generated': False,
                'confidence': 0.0,
                'analysis_details': {},
                'indicators': [],
                'caption': ''
            }
            
            # Method 1: Image characteristics analysis
            characteristics_score = self._analyze_image_characteristics(image)
            analysis_results['analysis_details']['characteristics'] = characteristics_score
            
            # Method 2: AI model analysis (if available)
            if self.model and self.processor:
                ai_analysis = self._analyze_with_ai_model(image)
                analysis_results['analysis_details']['ai_model'] = ai_analysis
                analysis_results['caption'] = ai_analysis.get('caption', '')
            
            # Method 3: Metadata analysis
            metadata_score = self._analyze_metadata(image_path)
            analysis_results['analysis_details']['metadata'] = metadata_score
            
            # Combine results
            final_result = self._combine_analysis_results(analysis_results)
            analysis_results.update(final_result)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in AI image detection: {e}")
            return {
                'error': str(e),
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': ['Analysis failed']
            }
    
    def _analyze_image_characteristics(self, image):
        """Analyze image characteristics that might indicate AI generation"""
        score = 0.0
        indicators = []
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Check for perfect symmetry (common in AI art)
        if self._check_symmetry(img_array):
            score += 0.3
            indicators.append("High symmetry detected")
        
        # Check for unusual color patterns
        color_analysis = self._analyze_colors(img_array)
        if color_analysis['unusual_patterns']:
            score += 0.2
            indicators.append("Unusual color patterns")
        
        # Check for perfect details (AI often creates overly perfect details)
        detail_analysis = self._analyze_details(img_array)
        if detail_analysis['overly_perfect']:
            score += 0.25
            indicators.append("Overly perfect details")
        
        # Check for common AI artifacts
        artifacts = self._check_ai_artifacts(img_array)
        if artifacts:
            score += 0.3
            indicators.extend(artifacts)
        
        return {
            'score': min(score, 1.0),
            'indicators': indicators
        }
    
    def _analyze_with_ai_model(self, image):
        """Use AI model to analyze the image"""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Look for AI-related keywords in caption
            ai_keywords = ['artificial', 'digital art', 'generated', 'synthetic', 'computer', 'rendered']
            ai_score = sum(1 for keyword in ai_keywords if keyword in caption.lower()) * 0.1
            
            return {
                'caption': caption,
                'ai_score': min(ai_score, 0.5),
                'indicators': [f"AI model analysis: {caption[:50]}..."]
            }
        except Exception as e:
            logger.error(f"AI model analysis failed: {e}")
            return {'caption': '', 'ai_score': 0.0, 'indicators': []}
    
    def _analyze_metadata(self, image_path):
        """Analyze image metadata for AI generation indicators"""
        score = 0.0
        indicators = []
        
        try:
            with Image.open(image_path) as img:
                # Check for common AI generation software in metadata
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    software = exif.get(271, '').lower() if exif else ''
                    
                    ai_software = ['midjourney', 'dall-e', 'stable diffusion', 'gpt', 'ai', 'generated']
                    if any(sw in software for sw in ai_software):
                        score += 0.4
                        indicators.append(f"AI software detected: {software}")
                
                # Check file size and format (AI images often have specific characteristics)
                file_size = os.path.getsize(image_path)
                if file_size < 100000:  # Very small files might be AI-generated
                    score += 0.1
                    indicators.append("Unusually small file size")
                
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
        
        return {
            'score': score,
            'indicators': indicators
        }
    
    def _check_symmetry(self, img_array):
        """Check for perfect symmetry which is common in AI art"""
        height, width = img_array.shape[:2]
        
        # Check horizontal symmetry
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:])
        
        if left_half.shape == right_half.shape:
            diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            return diff < 10  # Very low difference indicates high symmetry
        
        return False
    
    def _analyze_colors(self, img_array):
        """Analyze color patterns for AI generation indicators"""
        # Check for unusual color distributions
        colors = img_array.reshape(-1, 3)
        color_std = np.std(colors, axis=0)
        
        # AI images often have very specific color patterns
        unusual_patterns = np.any(color_std < 5) or np.any(color_std > 100)
        
        return {'unusual_patterns': unusual_patterns}
    
    def _analyze_details(self, img_array):
        """Analyze image details for AI generation indicators"""
        # Check for overly perfect details (AI often creates perfect textures)
        gray = np.mean(img_array, axis=2)
        detail_variance = np.var(gray)
        
        # Very high or very low variance might indicate AI generation
        overly_perfect = detail_variance < 100 or detail_variance > 10000
        
        return {'overly_perfect': overly_perfect}
    
    def _check_ai_artifacts(self, img_array):
        """Check for common AI generation artifacts"""
        artifacts = []
        
        # Check for common AI artifacts like:
        # - Perfect gradients
        # - Unusual edge patterns
        # - Specific noise patterns
        
        # Simple edge detection for artifacts
        gray = np.mean(img_array, axis=2)
        edges = np.abs(np.diff(gray, axis=1))
        edge_variance = np.var(edges)
        
        if edge_variance < 50:
            artifacts.append("Unusual edge patterns")
        
        return artifacts
    
    def _combine_analysis_results(self, analysis_results):
        """Combine all analysis results to determine if image is AI-generated"""
        total_score = 0.0
        all_indicators = []
        
        # Weight different analysis methods
        weights = {
            'characteristics': 0.4,
            'ai_model': 0.3,
            'metadata': 0.3
        }
        
        for method, weight in weights.items():
            if method in analysis_results['analysis_details']:
                method_result = analysis_results['analysis_details'][method]
                if 'score' in method_result:
                    total_score += method_result['score'] * weight
                if 'indicators' in method_result:
                    all_indicators.extend(method_result['indicators'])
        
        # Determine if AI-generated based on combined score
        is_ai_generated = total_score > 0.4
        confidence = min(total_score, 1.0)
        
        return {
            'is_ai_generated': is_ai_generated,
            'confidence': confidence,
            'indicators': all_indicators[:5]  # Limit to top 5 indicators
        }

# Global instance
detection_service = AIImageDetectionService()
