from django.db import models
from django.utils import timezone
import os

def upload_to(instance, filename):
    """Generate upload path for images"""
    return f'images/{filename}'

class ImageUpload(models.Model):
    image = models.ImageField(upload_to=upload_to)
    uploaded_at = models.DateTimeField(default=timezone.now)
    
    # AI Detection results
    is_ai_generated = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    detection_indicators = models.JSONField(default=list, blank=True)
    analysis_details = models.JSONField(default=dict, blank=True)
    ai_caption = models.TextField(blank=True)
    
    def __str__(self):
        ai_status = "AI-Generated" if self.is_ai_generated else "Real"
        return f"{ai_status} image uploaded at {self.uploaded_at}"
    
    @property
    def filename(self):
        return os.path.basename(self.image.name)
    
    @property
    def detection_status(self):
        """Return human-readable detection status"""
        if self.is_ai_generated:
            return "AI-Generated"
        return "Real/Human-Created"
    
    @property
    def confidence_percentage(self):
        """Return confidence as percentage"""
        return f"{self.confidence_score * 100:.1f}%"
