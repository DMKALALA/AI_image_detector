# Security Documentation

## Overview

This document outlines the security measures implemented in the AI Image Detector application, including file upload validation, API authentication, and CSRF protection.

## Security Features

### 1. File Upload Validation

All file uploads are validated using multiple layers of security:

- **File Size Limits**: Maximum 10MB per file
- **File Extension Validation**: Only allowed extensions: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- **MIME Type Validation**: Validates content type matches allowed image types
- **Image Content Verification**: Uses PIL to verify the file is a valid, non-corrupted image
- **Dimension Limits**: Prevents decompression bombs (max 10000x10000 pixels)
- **Format Validation**: Only JPEG, PNG, GIF, BMP, and WEBP formats are accepted

### 2. Filename Sanitization

All uploaded filenames are sanitized to prevent:
- Path traversal attacks (`../`, `..\\`)
- Directory separator injection (`/`, `\`)
- Special character injection
- Filename length overflow (max 255 characters)

Filenames are automatically sanitized and prefixed with timestamps to ensure uniqueness and prevent collisions.

### 3. API Authentication

All API endpoints require authentication via API key:

**Endpoints requiring API key:**
- `/api/detect/` - Single image detection
- `/api/detect/realtime/` - Real-time detection
- `/api/detect/batch/` - Batch detection

**How to use:**

1. **Set API key in environment:**
   ```bash
   export API_KEY=your-secret-api-key-here
   ```

2. **Or set in Django settings:**
   ```python
   API_KEY = 'your-secret-api-key-here'
   ```

3. **Include in API requests:**

   **Option 1: Header (Recommended)**
   ```bash
   curl -X POST \
     -H "X-API-Key: your-secret-api-key-here" \
     -F "image=@image.jpg" \
     http://localhost:8000/api/detect/
   ```

   **Option 2: Authorization Bearer**
   ```bash
   curl -X POST \
     -H "Authorization: Bearer your-secret-api-key-here" \
     -F "image=@image.jpg" \
     http://localhost:8000/api/detect/
   ```

**Development Mode:**
If no `API_KEY` is set, API endpoints will allow unauthenticated access (with a warning in logs). **This should never be used in production.**

### 4. CSRF Protection

- **Web Forms**: All web forms (home page upload, feedback submission) are protected by Django's CSRF middleware
- **API Endpoints**: Use API key authentication instead of CSRF tokens (appropriate for programmatic access)
- **Feedback Endpoint**: Removed `@csrf_exempt` - now properly protected with CSRF tokens

### 5. Error Handling

- Sensitive error details are not exposed to clients
- Generic error messages are returned to prevent information leakage
- Detailed errors are logged server-side for debugging

## Production Deployment Checklist

- [ ] Set `DEBUG=False` in production
- [ ] Set a strong `SECRET_KEY` (use environment variable)
- [ ] Set `API_KEY` environment variable for API authentication
- [ ] Configure `ALLOWED_HOSTS` properly
- [ ] Use HTTPS in production
- [ ] Configure proper file storage (e.g., S3, Azure Blob) instead of local filesystem
- [ ] Set up rate limiting for API endpoints
- [ ] Configure CORS properly if using from web applications
- [ ] Set up monitoring and alerting for security events
- [ ] Regularly update dependencies for security patches

## Security Utilities

The `detector/security_utils.py` module provides:

- `validate_image_file(file, check_content=True)`: Comprehensive file validation
- `sanitize_filename(filename)`: Filename sanitization
- `require_api_key(request)`: API key authentication check

## Example: Secure API Usage

```python
import requests

API_KEY = "your-secret-api-key"
API_URL = "https://your-domain.com/api/detect/"

headers = {
    "X-API-Key": API_KEY
}

files = {
    "image": ("image.jpg", open("image.jpg", "rb"), "image/jpeg")
}

response = requests.post(API_URL, headers=headers, files=files)
result = response.json()

if result.get("success"):
    print(f"AI Generated: {result['is_ai_generated']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Error: {result['error']}")
```

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:
1. Do not open a public issue
2. Contact the maintainers directly
3. Provide detailed information about the vulnerability
4. Allow time for a fix before public disclosure

