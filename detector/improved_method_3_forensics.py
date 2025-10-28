"""
Improved Method 3: Advanced Image Forensics Analysis
Enhanced with additional techniques for higher accuracy and decisive tie-breaking:
- Error Level Analysis (ELA) - detects compression artifacts
- Noise pattern analysis - AI images have different noise characteristics
- Color space analysis - JPEG compression inconsistencies  
- DCT coefficient analysis - frequency domain patterns
- CFA (Color Filter Array) pattern analysis - new technique
- Gradient consistency analysis - new technique
"""

import numpy as np
import cv2
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple
from scipy import fft
from scipy.fftpack import dct
import os

logger = logging.getLogger(__name__)

class ImprovedForensicsMethod3:
    """
    Improved Method 3 using advanced image forensics techniques
    Based on research in digital image forensics and synthetic image detection
    Enhanced to be decisive tie-breaker when Methods 1 & 2 disagree
    """
    
    def __init__(self):
        self.ai_indicators = []
        self.real_indicators = []
        logger.info("Improved Method 3 (Advanced Forensics) initialized")
    
    def detect(self, image_path: str) -> dict:
        """
        Detect if image is AI-generated using advanced forensics techniques
        
        Returns:
            dict with 'is_ai_generated', 'confidence', 'indicators'
        """
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            # Perform multiple forensic analyses
            ela_score = self._error_level_analysis(image_path)
            noise_score = self._noise_pattern_analysis(img_array)
            color_artifacts = self._color_space_analysis(img_array)
            dct_analysis = self._dct_coefficient_analysis(img_array)
            cfa_pattern = self._cfa_pattern_analysis(img_array)  # New technique
            gradient_consistency = self._gradient_consistency_analysis(img_array)  # New technique
            
            # Combine scores with improved weighting
            ai_score = 0.0
            factors = []
            indicators = []
            
            # Error Level Analysis (ELA) - very reliable for compression artifacts
            if ela_score > 0.6:  # High ELA score suggests AI
                weight = 0.30  # Strong weight - ELA is reliable
                ai_score += weight
                factors.append('high_ela_score')
                indicators.append(f"High error level analysis score ({ela_score:.2f}) - suggests AI-generated compression patterns (weight: {weight:.2f})")
            elif ela_score < 0.2:  # Low ELA suggests real photo
                weight = -0.15  # Negative weight for real
                ai_score += weight
                factors.append('low_ela_score')
                indicators.append(f"Low error level analysis score ({ela_score:.2f}) - suggests authentic compression (weight: {abs(weight):.2f})")
            
            # Noise pattern analysis
            if noise_score > 0.65:  # AI images often have uniform noise
                weight = 0.25
                ai_score += weight
                factors.append('uniform_noise')
                indicators.append(f"Uniform noise pattern detected (score: {noise_score:.2f}) - characteristic of AI generation (weight: {weight:.2f})")
            elif noise_score < 0.35:  # Real photos have natural noise variation
                weight = -0.10
                ai_score += weight
                factors.append('natural_noise')
                indicators.append(f"Natural noise variation detected (score: {noise_score:.2f}) - suggests real photo (weight: {abs(weight):.2f})")
            
            # Color space artifacts
            if color_artifacts > 0.7:  # Color inconsistencies
                weight = 0.20
                ai_score += weight
                factors.append('color_artifacts')
                indicators.append(f"Color space inconsistencies detected (score: {color_artifacts:.2f}) - possible AI artifacts (weight: {weight:.2f})")
            elif color_artifacts < 0.3:  # Clean color transitions
                weight = -0.10
                ai_score += weight
                factors.append('clean_color')
                indicators.append(f"Clean color transitions detected (score: {color_artifacts:.2f}) - suggests real photo (weight: {abs(weight):.2f})")
            
            # DCT coefficient analysis
            if dct_analysis > 0.6:  # Irregular DCT patterns suggest manipulation
                weight = 0.25
                ai_score += weight
                factors.append('irregular_dct')
                indicators.append(f"Irregular DCT coefficient patterns (score: {dct_analysis:.2f}) - suggests AI manipulation (weight: {weight:.2f})")
            elif dct_analysis < 0.4:  # Regular DCT patterns
                weight = -0.10
                ai_score += weight
                factors.append('regular_dct')
                indicators.append(f"Regular DCT coefficient patterns (score: {dct_analysis:.2f}) - suggests natural image (weight: {abs(weight):.2f})")
            
            # NEW: CFA (Color Filter Array) pattern analysis
            # Real cameras use Bayer filters - AI images may lack proper CFA patterns
            if cfa_pattern > 0.65:  # Weak CFA pattern suggests AI
                weight = 0.20
                ai_score += weight
                factors.append('weak_cfa_pattern')
                indicators.append(f"Weak Color Filter Array pattern (score: {cfa_pattern:.2f}) - suggests AI generation (weight: {weight:.2f})")
            elif cfa_pattern < 0.35:  # Strong CFA pattern suggests real camera
                weight = -0.12
                ai_score += weight
                factors.append('strong_cfa_pattern')
                indicators.append(f"Strong Color Filter Array pattern (score: {cfa_pattern:.2f}) - suggests real camera (weight: {abs(weight):.2f})")
            
            # NEW: Gradient consistency analysis
            # Real photos have more consistent edge gradients - AI may show inconsistencies
            if gradient_consistency < 0.4:  # Low consistency suggests AI
                weight = 0.18
                ai_score += weight
                factors.append('inconsistent_gradients')
                indicators.append(f"Inconsistent gradient patterns (score: {gradient_consistency:.2f}) - suggests AI generation (weight: {weight:.2f})")
            elif gradient_consistency > 0.65:  # High consistency suggests real
                weight = -0.10
                ai_score += weight
                factors.append('consistent_gradients')
                indicators.append(f"Consistent gradient patterns (score: {gradient_consistency:.2f}) - suggests real photo (weight: {abs(weight):.2f})")
            
            # Normalize score to [0, 1] range
            # Maximum positive score: 1.38, minimum negative score: -0.67
            # Normalize: (score + 0.67) / 2.05
            normalized_score = max(0.0, min(1.0, (ai_score + 0.67) / 2.05))
            
            # Improved threshold logic - more decisive when Methods 1 & 2 disagree
            # Based on research: Method 3 should be decisive tie-breaker
            threshold = 0.32  # Slightly more sensitive
            
            # If we have strong indicators, be more decisive
            strong_indicators = sum([
                1 for factor in factors 
                if factor in ['high_ela_score', 'irregular_dct', 'uniform_noise', 'weak_cfa_pattern']
            ])
            
            # Boost confidence if multiple strong indicators agree
            if strong_indicators >= 2:
                threshold = 0.30  # Lower threshold when multiple indicators agree
                confidence_boost = 1.15
            elif strong_indicators == 1 and len(factors) >= 2:
                threshold = 0.31
                confidence_boost = 1.10
            else:
                confidence_boost = 1.0
            
            is_ai_generated = normalized_score > threshold
            confidence = min(1.0, abs(normalized_score - 0.5) * 2 * confidence_boost)
            
            # If no factors detected, make a conservative call
            if not factors:
                indicators.append("No strong forensic indicators detected")
                # Don't default - use score-based decision
                is_ai_generated = normalized_score > 0.45  # Slightly more aggressive default
                confidence = max(0.25, normalized_score * 0.6)  # Lower confidence when uncertain
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'score': normalized_score,
                'factors': factors,
                'indicators': indicators,
                'analysis_details': {
                    'ela_score': ela_score,
                    'noise_score': noise_score,
                    'color_artifacts': color_artifacts,
                    'dct_analysis': dct_analysis,
                    'cfa_pattern': cfa_pattern,
                    'gradient_consistency': gradient_consistency
                }
            }
            
        except Exception as e:
            logger.error(f"Error in improved Method 3 detection: {e}", exc_info=True)
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'error': str(e),
                'indicators': [f'Improved forensics detection error: {str(e)}']
            }
    
    def _error_level_analysis(self, image_path: str) -> float:
        """
        Error Level Analysis (ELA) - detects compression artifacts
        AI-generated images often show different compression patterns when re-saved
        Returns score 0-1 (higher = more likely AI)
        """
        try:
            # Read original image
            with open(image_path, 'rb') as f:
                original_data = f.read()
            
            # Save image at JPEG quality 90 and compare
            img = Image.open(io.BytesIO(original_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to memory at different quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            recompressed = Image.open(buffer)
            
            # Convert to numpy arrays
            original = np.array(img, dtype=np.float32)
            recompressed_array = np.array(recompressed, dtype=np.float32)
            
            # Resize recompressed to match original if needed
            if original.shape != recompressed_array.shape:
                recompressed_array = cv2.resize(recompressed_array, (original.shape[1], original.shape[0]))
            
            # Calculate error level
            error = np.abs(original - recompressed_array)
            error_level = np.mean(error)
            
            # Normalize to 0-1 scale
            # AI images typically have error_level between 2-8
            # Real photos typically have error_level between 1-3
            normalized = min(1.0, error_level / 10.0)
            
            return float(normalized)
            
        except Exception as e:
            logger.warning(f"ELA analysis error: {e}")
            return 0.5  # Neutral score on error
    
    def _noise_pattern_analysis(self, img_array: np.ndarray) -> float:
        """
        Analyze noise patterns - AI images often have uniform noise
        Returns score 0-1 (higher = more uniform noise = more likely AI)
        """
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply high-pass filter to extract noise
            kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
            noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # Calculate noise statistics
            noise_std = np.std(noise)
            noise_var = np.var(noise)
            
            # Calculate noise uniformity (coefficient of variation)
            # Lower variation = more uniform noise = more likely AI
            if noise_std > 0:
                cv_value = noise_std / (np.abs(np.mean(noise)) + 1e-8)
                # Inverse relationship: lower CV = higher score
                uniformity_score = 1.0 / (1.0 + cv_value * 10)
            else:
                uniformity_score = 0.5
            
            return float(uniformity_score)
            
        except Exception as e:
            logger.warning(f"Noise analysis error: {e}")
            return 0.5
    
    def _color_space_analysis(self, img_array: np.ndarray) -> float:
        """
        Analyze color space for artifacts and inconsistencies
        AI images may show color banding or unnatural transitions
        Returns score 0-1 (higher = more artifacts = more likely AI)
        """
        try:
            # Convert to LAB color space (better for perceptual analysis)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            a_channel = lab[:, :, 1].astype(np.float32)
            b_channel = lab[:, :, 2].astype(np.float32)
            
            # Calculate gradient magnitude in color space
            grad_a_x = cv2.Sobel(a_channel, cv2.CV_32F, 1, 0, ksize=3)
            grad_a_y = cv2.Sobel(a_channel, cv2.CV_32F, 0, 1, ksize=3)
            grad_b_x = cv2.Sobel(b_channel, cv2.CV_32F, 1, 0, ksize=3)
            grad_b_y = cv2.Sobel(b_channel, cv2.CV_32F, 0, 1, ksize=3)
            
            grad_magnitude_a = np.sqrt(grad_a_x**2 + grad_a_y**2)
            grad_magnitude_b = np.sqrt(grad_b_x**2 + grad_b_y**2)
            
            # Look for abrupt changes (artifacts) vs smooth transitions
            # AI images often have sudden color jumps
            threshold_a = np.percentile(grad_magnitude_a, 95)
            threshold_b = np.percentile(grad_magnitude_b, 95)
            
            abrupt_changes_a = np.sum(grad_magnitude_a > threshold_a) / grad_magnitude_a.size
            abrupt_changes_b = np.sum(grad_magnitude_b > threshold_b) / grad_magnitude_b.size
            
            artifact_score = (abrupt_changes_a + abrupt_changes_b) / 2.0
            
            return float(min(1.0, artifact_score * 5))  # Scale appropriately
            
        except Exception as e:
            logger.warning(f"Color space analysis error: {e}")
            return 0.5
    
    def _dct_coefficient_analysis(self, img_array: np.ndarray) -> float:
        """
        Analyze DCT (Discrete Cosine Transform) coefficients
        AI images may show different frequency patterns
        Returns score 0-1 (higher = more irregular patterns = more likely AI)
        """
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Divide into 8x8 blocks (standard JPEG block size)
            h, w = gray.shape
            blocks_h = h // 8
            blocks_w = w // 8
            
            # Crop to multiple of 8
            gray = gray[:blocks_h*8, :blocks_w*8]
            
            # Calculate DCT for each block
            dct_energies = []
            for i in range(blocks_h):
                for j in range(blocks_w):
                    block = gray[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                    # Subtract 128 (JPEG normalization)
                    block -= 128
                    # DCT transform
                    dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                    # Calculate energy (sum of squares)
                    energy = np.sum(dct_block**2)
                    dct_energies.append(energy)
            
            # Analyze distribution
            if len(dct_energies) > 1:
                # AI images may have more irregular energy distribution
                energy_std = np.std(dct_energies)
                energy_mean = np.mean(dct_energies)
                
                if energy_mean > 0:
                    cv_value = energy_std / energy_mean
                    # Higher CV = more irregular = higher score
                    irregularity_score = min(1.0, cv_value / 2.0)
                else:
                    irregularity_score = 0.5
            else:
                irregularity_score = 0.5
            
            return float(irregularity_score)
            
        except Exception as e:
            logger.warning(f"DCT analysis error: {e}")
            return 0.5
    
    def _cfa_pattern_analysis(self, img_array: np.ndarray) -> float:
        """
        Analyze Color Filter Array (CFA) patterns
        Real cameras use Bayer CFA patterns - AI images may lack proper patterns
        Returns score 0-1 (higher = weaker CFA pattern = more likely AI)
        """
        try:
            # Convert to grayscale for CFA analysis
            if len(img_array.shape) == 3:
                # Analyze each channel's pattern
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            else:
                return 0.5  # Can't analyze CFA on grayscale
            
            # Check for Bayer pattern characteristics
            # Real cameras show periodicity in color channels
            # Compute autocorrelation to detect periodic patterns
            def autocorrelation(channel):
                channel_norm = channel.astype(np.float32) - np.mean(channel)
                fft_channel = fft.fft2(channel_norm)
                autocorr = np.abs(fft.ifft2(fft_channel * np.conj(fft_channel)))
                # Look for periodicity at 2x2 (Bayer pattern size)
                autocorr_shifted = fft.fftshift(autocorr)
                center_h, center_w = autocorr_shifted.shape[0] // 2, autocorr_shifted.shape[1] // 2
                # Sample pattern around center
                pattern_strength = np.sum(autocorr_shifted[center_h-5:center_h+5, center_w-5:center_w+5])
                total_energy = np.sum(autocorr_shifted)
                return pattern_strength / (total_energy + 1e-8)
            
            r_pattern = autocorrelation(r)
            g_pattern = autocorrelation(g)
            b_pattern = autocorrelation(b)
            
            avg_pattern = (r_pattern + g_pattern + b_pattern) / 3.0
            
            # Invert: strong pattern (real) = low score, weak pattern (AI) = high score
            cfa_score = 1.0 - min(1.0, avg_pattern * 2.0)
            
            return float(cfa_score)
            
        except Exception as e:
            logger.warning(f"CFA analysis error: {e}")
            return 0.5
    
    def _gradient_consistency_analysis(self, img_array: np.ndarray) -> float:
        """
        Analyze gradient consistency across the image
        Real photos have more consistent edge gradients - AI may show inconsistencies
        Returns score 0-1 (higher = more consistent = more likely real)
        """
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_direction = np.arctan2(grad_y, grad_x)
            
            # Analyze consistency: divide into blocks and compare gradient distributions
            h, w = gray.shape
            block_size = 32
            blocks_h = h // block_size
            blocks_w = w // block_size
            
            if blocks_h < 2 or blocks_w < 2:
                return 0.5  # Too small for analysis
            
            block_directions = []
            for i in range(blocks_h):
                for j in range(blocks_w):
                    block_dir = grad_direction[i*block_size:(i+1)*block_size, 
                                             j*block_size:(j+1)*block_size]
                    # Only consider strong edges
                    block_mag = grad_magnitude[i*block_size:(i+1)*block_size,
                                             j*block_size:(j+1)*block_size]
                    strong_edges = block_dir[block_mag > np.percentile(block_mag, 80)]
                    if len(strong_edges) > 0:
                        # Compute circular variance (consistency measure)
                        mean_dir = np.mean(strong_edges)
                        variance = np.var(strong_edges)
                        consistency = 1.0 / (1.0 + variance)  # Higher variance = lower consistency
                        block_directions.append(consistency)
            
            if len(block_directions) > 0:
                consistency_score = np.mean(block_directions)
            else:
                consistency_score = 0.5
            
            return float(consistency_score)
            
        except Exception as e:
            logger.warning(f"Gradient consistency analysis error: {e}")
            return 0.5
