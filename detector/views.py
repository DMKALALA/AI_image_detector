from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from .models import ImageUpload
from .forms import ImageUploadForm
from .ai_service import detection_service
import os

def home(request):
    """Home page with image upload form"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save()
            
            # Perform AI detection
            try:
                result = detection_service.detect_ai_image(image_upload.image.path)
                
                if 'error' not in result:
                    image_upload.is_ai_generated = result['is_ai_generated']
                    image_upload.confidence_score = result['confidence']
                    image_upload.detection_indicators = result['indicators']
                    image_upload.analysis_details = result['analysis_details']
                    image_upload.ai_caption = result.get('caption', '')
                    image_upload.save()
                    
                    status = "AI-Generated" if result['is_ai_generated'] else "Real/Human-Created"
                    messages.success(request, f'Image analyzed! Result: {status} (Confidence: {result["confidence"]*100:.1f}%)')
                else:
                    messages.error(request, f'Detection failed: {result["error"]}')
                    
            except Exception as e:
                messages.error(request, f'Error processing image: {str(e)}')
            
            return redirect('detector:result', pk=image_upload.pk)
    else:
        form = ImageUploadForm()
    
    # Get recent uploads
    recent_uploads = ImageUpload.objects.all().order_by('-uploaded_at')[:5]
    
    return render(request, 'detector/home.html', {
        'form': form,
        'recent_uploads': recent_uploads
    })

def result(request, pk):
    """Display detection results"""
    try:
        image_upload = ImageUpload.objects.get(pk=pk)
        return render(request, 'detector/result.html', {
            'image_upload': image_upload
        })
    except ImageUpload.DoesNotExist:
        messages.error(request, 'Image not found')
        return redirect('detector:home')

def api_detect(request):
    """API endpoint for AI image detection"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save()
            
            try:
                result = detection_service.detect_ai_image(image_upload.image.path)
                
                if 'error' not in result:
                    image_upload.is_ai_generated = result['is_ai_generated']
                    image_upload.confidence_score = result['confidence']
                    image_upload.detection_indicators = result['indicators']
                    image_upload.analysis_details = result['analysis_details']
                    image_upload.ai_caption = result.get('caption', '')
                    image_upload.save()
                
                return JsonResponse({
                    'success': True,
                    'image_id': image_upload.pk,
                    'is_ai_generated': result.get('is_ai_generated', False),
                    'confidence': result.get('confidence', 0.0),
                    'indicators': result.get('indicators', []),
                    'caption': result.get('caption', ''),
                    'error': result.get('error', None)
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request'
    })
