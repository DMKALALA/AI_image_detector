from django.contrib import admin
from .models import ImageUpload

@admin.register(ImageUpload)
class ImageUploadAdmin(admin.ModelAdmin):
    list_display = ['filename', 'uploaded_at', 'detection_status', 'confidence_percentage']
    list_filter = ['uploaded_at', 'is_ai_generated']
    search_fields = ['image', 'ai_caption']
    readonly_fields = ['uploaded_at', 'confidence_percentage']
    
    def detection_status(self, obj):
        return obj.detection_status
    detection_status.short_description = 'AI Detection'
