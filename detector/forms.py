from django import forms
from django.core.exceptions import ValidationError
from .models import ImageUpload
from .security_utils import validate_image_file, sanitize_filename

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            })
        }
    
    def clean_image(self):
        """Validate uploaded image file"""
        image = self.cleaned_data.get('image')
        if not image:
            return image
        
        # Validate file
        is_valid, error_msg = validate_image_file(image)
        if not is_valid:
            raise ValidationError(error_msg)
        
        # Sanitize filename
        if hasattr(image, 'name'):
            image.name = sanitize_filename(image.name)
        
        return image
