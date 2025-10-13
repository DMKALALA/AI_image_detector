# AI Image Detector

A Django web application that uses artificial intelligence to detect whether images are AI-generated or real/human-created. Built with Django, PyTorch, and the BLIP (Bootstrapping Language-Image Pre-training) model from Hugging Face Transformers.

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

5. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

6. **Open your browser** and go to `http://127.0.0.1:8000/`

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

- `GET /` - Home page with upload form
- `POST /` - Upload and analyze image
- `GET /result/<id>/` - View detection results
- `POST /api/detect/` - API endpoint for programmatic access

## How It Works

The AI detection system uses multiple analysis methods:

1. **Image Characteristics Analysis**: 
   - Checks for perfect symmetry (common in AI art)
   - Analyzes unusual color patterns
   - Detects overly perfect details
   - Identifies common AI generation artifacts

2. **AI Model Analysis**: 
   - Uses BLIP model to generate image captions
   - Analyzes content for AI-related keywords
   - Provides contextual understanding

3. **Metadata Analysis**: 
   - Examines file metadata for AI software signatures
   - Checks file size and format characteristics
   - Looks for generation tool indicators

4. **Combined Scoring**: 
   - Weights different analysis methods
   - Provides confidence scores
   - Returns specific detection indicators

## Model Information

The application uses the **BLIP (Bootstrapping Language-Image Pre-training)** model:
- **Purpose**: Image captioning and visual question answering
- **Provider**: Salesforce Research
- **Framework**: Hugging Face Transformers
- **Capabilities**: Understands both visual and textual information

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

## Security Considerations

- This is a development setup - not suitable for production without additional security measures
- File uploads should be validated and sanitized in production
- Consider implementing user authentication for multi-user scenarios
- Set up proper file storage (e.g., AWS S3) for production deployments

## Future Enhancements

- [ ] User authentication and profiles
- [ ] Batch image processing
- [ ] More sophisticated AI detection models
- [ ] Real-time image processing with WebSockets
- [ ] API rate limiting
- [ ] Cloud storage integration
- [ ] Advanced artifact detection

## License

This project is for educational purposes. Please ensure you comply with the licenses of the underlying models and libraries used.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application!
