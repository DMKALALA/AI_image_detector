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
    method = models.CharField(max_length=100, default='unknown')
    
    # User feedback system
    user_feedback = models.CharField(
        max_length=20,
        choices=[
            ('correct', 'Correct'),
            ('incorrect', 'Incorrect'),
            ('unsure', 'Unsure'),
            ('', 'No Feedback')
        ],
        default='',
        blank=True
    )
    feedback_notes = models.TextField(blank=True)
    feedback_timestamp = models.DateTimeField(null=True, blank=True)
    
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
    
    def save_feedback(self, feedback, notes=''):
        """Save user feedback for this image"""
        self.user_feedback = feedback
        self.feedback_notes = notes
        self.feedback_timestamp = timezone.now()
        self.save()
    
    def get_feedback_status(self):
        """Get feedback status for display"""
        if self.user_feedback == 'correct':
            return '✅ Correct'
        elif self.user_feedback == 'incorrect':
            return '❌ Incorrect'
        elif self.user_feedback == 'unsure':
            return '❓ Unsure'
        else:
            return '📝 No Feedback'

class MethodPerformance(models.Model):
    """Track performance of different detection methods"""
    method_name = models.CharField(max_length=50, unique=True)
    total_detections = models.IntegerField(default=0)
    correct_detections = models.IntegerField(default=0)
    incorrect_detections = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    
    @property
    def accuracy(self):
        """Calculate accuracy percentage"""
        if self.total_detections == 0:
            return 0.0
        return (self.correct_detections / self.total_detections) * 100
    
    @property
    def accuracy_decimal(self):
        """Calculate accuracy as decimal"""
        if self.total_detections == 0:
            return 0.0
        return self.correct_detections / self.total_detections
    
    def __str__(self):
        return f"{self.method_name}: {self.accuracy:.1f}% accuracy ({self.correct_detections}/{self.total_detections})"
    
    class Meta:
        verbose_name = "Method Performance"
        verbose_name_plural = "Method Performances"
