# AI Image Detector

A production-ready Django web application that uses an ensemble of 5 advanced AI detection methods to identify whether images are AI-generated or real/human-created. Built with Django, PyTorch, and multiple state-of-the-art models from Hugging Face Transformers.

**Status**: ✅ Production-ready with comprehensive security hardening

## 🌐 Live Demo

**Deployed Application**: https://ai-image-detector-1.onrender.com

> **Note**: Free instances on Render spin down after periods of inactivity. First request may take 30-50 seconds to wake up the service.

## Features

- 🖼️ **Image Upload**: Upload images in various formats (JPG, PNG, GIF, WebP)
- 🤖 **AI Detection**: Automatic detection of AI-generated vs real images using state-of-the-art machine learning models
- 📊 **Confidence Scores**: See how confident the AI is about each detection
- 📱 **Responsive Design**: Modern, mobile-friendly interface
- 🔄 **Recent Uploads**: View your recent image analyses
- 🎯 **Real-time Processing**: Fast AI-powered analysis
- 🔍 **Detailed Analysis**: Multiple detection methods with specific indicators

## Technology Stack

- **Backend**: Django 4.2.7
- **AI/ML**: PyTorch, Transformers (Hugging Face)
- **Model**: BLIP (Salesforce/blip-image-captioning-base)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Database**: SQLite (development)
- **Image Processing**: Pillow (PIL), NumPy

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DMKALALA/AI_image_detector.git
   cd AI_image_detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run database migrations**:
   ```bash
   python manage.py migrate
   ```

4. **Create a superuser (optional)**:
   ```bash
   python manage.py createsuperuser
   ```

5. **Set environment variables** (create a `.env` file or export):
   ```bash
   export SECRET_KEY=your-secret-key-here
   export DEBUG=True  # Set to False for production
   export ALLOWED_HOSTS=localhost,127.0.0.1
   export API_KEY=your-api-key-here  # Optional for development
   ```

6. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

7. **Open your browser** and go to `http://127.0.0.1:8000/`

**Note:** For normal operation, leave `ENABLE_MODEL_IMPORTS` unset or set to `1` (default) so PyTorch models load properly.

## Usage

1. **Upload an Image**: 
   - Click on the upload area or drag and drop an image
   - Supported formats: JPG, PNG, GIF, WebP

2. **View AI Detection Results**:
   - The AI will analyze your image and determine if it's AI-generated or real
   - See confidence scores and detailed indicators
   - View technical details about the analysis

3. **Admin Interface**:
   - Visit `http://127.0.0.1:8000/admin/` to manage uploaded images
   - Use the superuser account you created

## API Endpoints

All API endpoints require authentication via API key. Set `API_KEY` environment variable.

- `GET /` - Home page with upload form
- `POST /` - Upload and analyze image (web form, CSRF protected)
- `GET /result/<id>/` - View detection results
- `POST /api/detect/` - API endpoint for programmatic access (requires API key)
- `POST /api/detect/realtime/` - Real-time detection API (requires API key)
- `POST /api/detect/batch/` - Batch detection API (max 10 images, requires API key)
- `GET /api/status/` - API status and health check
- `GET /api/stats/` - Detection statistics
- `POST /feedback/<image_id>/` - Submit user feedback (CSRF protected)

**API Authentication:**
```bash
# Set API key
export API_KEY=your-secret-api-key

# Use in requests
curl -X POST \
  -H "X-API-Key: your-secret-api-key" \
  -F "image=@image.jpg" \
  http://localhost:8000/api/detect/
```

## How It Works

The AI detection system uses **5 advanced detection methods** with weighted ensemble voting:

1. **Method 1: Deep Learning (EfficientNet/ViT/ResNet)**
   - State-of-the-art neural network classifiers
   - Trained on large-scale AI/real image datasets
   - High accuracy for modern AI generators

2. **Method 2: Statistical Pattern Analysis**
   - Analyzes pixel-level statistical patterns
   - Detects frequency domain artifacts
   - Identifies compression inconsistencies

3. **Method 3: Advanced Spectral & Forensics**
   - Frequency domain analysis (FFT, DCT)
   - Detects AI generation artifacts
   - Metadata and file structure analysis

4. **Method 4: HuggingFace Specialist Models**
   - Vision Transformer AI-image-detector
   - AI vs Human Image Detector
   - Fine-tuned on GenImage dataset

5. **Method 5: Enterprise-Grade Models**
   - Hive-style CNN classifiers
   - Reality Defender-style transformers
   - CLIP-based semantic inconsistency detection

**Ensemble Voting:**
- Weighted voting combines all 5 methods
- Confidence calibration for each method
- Adaptive thresholds based on agreement
- Feedback-based learning for continuous improvement

## Model Information

The application uses an **ensemble of 5 detection methods**:

- **Method 1**: Deep Learning models (EfficientNet, Vision Transformers, ResNet)
- **Method 2**: Statistical pattern analysis algorithms
- **Method 3**: Advanced spectral and forensic analysis
- **Method 4**: HuggingFace specialist models (ViT AI-detector, AI/Human classifier)
- **Method 5**: Enterprise-grade models (Hive-style, Reality Defender-style, CLIP-based)

All models are fine-tuned on the GenImage dataset for optimal performance. See `docs/FINE_TUNING_GUIDE.md` for training details.

## File Structure

```
AI_image_detector/
├── detector/                 # Main Django app
│   ├── models.py            # Database models
│   ├── views.py             # View functions
│   ├── forms.py             # Django forms
│   ├── urls.py              # URL patterns
│   ├── admin.py             # Admin configuration
│   ├── ai_service.py        # AI detection service
│   └── templates/           # HTML templates
│       └── detector/
│           ├── base.html    # Base template
│           ├── home.html    # Home page
│           └── result.html  # Results page
├── image_detector_project/  # Django project settings
│   ├── settings.py          # Project configuration
│   ├── urls.py              # Main URL configuration
│   └── wsgi.py              # WSGI configuration
├── media/                   # Uploaded files (created automatically)
├── static/                  # Static files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Documentation

For comprehensive project documentation including architecture, performance analysis, training guides, and development history, see **`PROJECT_DOCUMENTATION.md`**.

## Customization

### Modifying Detection Methods
Edit `detector/ai_service.py` to adjust detection algorithms and thresholds.

### Changing the AI Model
Replace the model in `detector/ai_service.py`:
```python
# Change this line to use a different model
self.model = BlipForConditionalGeneration.from_pretrained("your-model-name")
```

### Styling
Modify the CSS in `detector/templates/detector/base.html` to customize the appearance.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure you have enough disk space (model is ~1GB)
   - Check your internet connection for model download
   - Verify PyTorch installation

2. **Memory Issues**:
   - The model requires significant RAM
   - Consider using CPU-only mode if GPU memory is insufficient

3. **Upload Errors**:
   - Check file size limits
   - Ensure supported image formats
   - Verify write permissions for media directory

## Performance Notes

- **First Run**: Model download and initialization may take a few minutes
- **Processing Time**: Image analysis typically takes 2-5 seconds per image
- **Memory Usage**: ~2-4GB RAM recommended for smooth operation
- **Storage**: Each uploaded image is stored locally in the `media/` directory

## Security Features

✅ **Production-Ready Security Implemented:**

- ✅ **File Upload Validation**: Comprehensive validation (size, MIME type, extension, content verification)
- ✅ **Filename Sanitization**: Prevents path traversal and injection attacks
- ✅ **API Authentication**: All API endpoints require API key authentication
- ✅ **CSRF Protection**: Web forms protected with Django CSRF middleware
- ✅ **Image Content Verification**: PIL-based validation prevents corrupted/malicious files
- ✅ **Dimension Limits**: Prevents decompression bomb attacks
- ✅ **Error Handling**: Secure error messages without information leakage

**Production Setup:**
1. Set `DEBUG=False` in production
2. Set `SECRET_KEY` environment variable (required)
3. Set `API_KEY` environment variable for API authentication
4. Configure `ALLOWED_HOSTS` properly (comma-separated list)
5. Use HTTPS in production
6. Configure proper file storage (e.g., AWS S3, Azure Blob)
7. Leave `ENABLE_MODEL_IMPORTS` unset or set to `1` (default) to enable PyTorch models
8. Install whitenoise for static file serving: `pip install whitenoise` (already in requirements.txt)

**Example Production Environment:**
```bash
export SECRET_KEY=your-strong-secret-key-here
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
export API_KEY=your-api-key-here
# ENABLE_MODEL_IMPORTS defaults to 1 (enabled) - leave unset for normal operation
```

See `docs/SECURITY.md` for complete security documentation.

## Features & Capabilities

✅ **Implemented:**
- ✅ 5-method ensemble detection system
- ✅ Batch image processing (up to 10 images)
- ✅ API endpoints with authentication
- ✅ User feedback system
- ✅ Adaptive learning from feedback
- ✅ Comprehensive security hardening
- ✅ Production-ready file validation
- ✅ Analytics dashboard
- ✅ Fine-tuning support for all models

**Future Enhancements:**
- [ ] User authentication and profiles
- [ ] API rate limiting
- [ ] Cloud storage integration (S3, Azure Blob)
- [ ] WebSocket-based real-time processing
- [ ] Advanced provenance detection (C2PA, SynthID)
- [ ] Model versioning and A/B testing

## License

This project is for educational purposes. Please ensure you comply with the licenses of the underlying models and libraries used.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application!
