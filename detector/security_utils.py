"""
Security utilities for file uploads and API authentication
"""
import os
import re
import hashlib
from typing import Tuple, Optional
from django.conf import settings
from django.http import JsonResponse
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
    'image/bmp', 'image/webp'
}

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection attacks.
    Returns a safe filename with only alphanumeric, dots, hyphens, and underscores.
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove any directory separators
    filename = filename.replace('/', '').replace('\\', '')
    
    # Keep only safe characters: alphanumeric, dots, hyphens, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Remove leading dots and ensure it's not empty
    filename = filename.lstrip('.')
    if not filename:
        filename = 'uploaded_image'
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def validate_image_file(file, check_content: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive image file validation.
    Returns (is_valid, error_message)
    """
    # Check file size
    if hasattr(file, 'size'):
        if file.size > MAX_FILE_SIZE:
            return False, f'File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB'
        if file.size == 0:
            return False, 'File is empty'
    
    # Check file extension
    filename = file.name if hasattr(file, 'name') else str(file)
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f'Invalid file extension. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
    
    # Check MIME type if available
    if hasattr(file, 'content_type'):
        if file.content_type not in ALLOWED_MIME_TYPES:
            return False, f'Invalid MIME type. Allowed: {", ".join(ALLOWED_MIME_TYPES)}'
    
    # Validate image content
    if check_content:
        try:
            file.seek(0)
            image = Image.open(file)
            image.verify()  # Verify it's a valid image
            
            # Reopen for actual use (verify() closes the file)
            file.seek(0)
            image = Image.open(file)
            
            # Check image dimensions (prevent decompression bombs)
            width, height = image.size
            if width > 10000 or height > 10000:
                return False, 'Image dimensions too large (max 10000x10000)'
            if width == 0 or height == 0:
                return False, 'Invalid image dimensions'
            
            # Check file format
            if image.format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']:
                return False, f'Unsupported image format: {image.format}'
            
            file.seek(0)
            
        except Exception as e:
            logger.warning(f"Image validation failed: {e}")
            return False, f'Invalid or corrupted image file: {str(e)}'
    
    return True, None

def require_api_key(request) -> Tuple[bool, Optional[JsonResponse]]:
    """
    Check for API key authentication.
    Returns (is_authenticated, error_response)
    """
    # Get API key from environment or settings
    expected_api_key = os.environ.get('API_KEY') or getattr(settings, 'API_KEY', None)
    
    # If no API key is configured, allow access (development mode)
    if not expected_api_key:
        logger.warning("API_KEY not configured - allowing unauthenticated access")
        return True, None
    
    # Check for API key in header
    api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
    
    if not api_key:
        return False, JsonResponse({
            'error': 'API key required. Provide X-API-Key header or Authorization: Bearer <key>'
        }, status=401)
    
    if api_key != expected_api_key:
        return False, JsonResponse({
            'error': 'Invalid API key'
        }, status=403)
    
    return True, None

