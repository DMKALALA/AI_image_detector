from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import ImageUpload
from .forms import ImageUploadForm
from .three_method_detection_service import get_detection_service
from .security_utils import validate_image_file, sanitize_filename, require_api_key
import os
import json
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def home(request):
    """Home page with image upload form"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save()
            
            # Perform AI detection using comparative service
            try:
                detection_service = get_detection_service()
                if detection_service is None:
                    messages.error(request, 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).')
                    return redirect('detector:home')
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

@csrf_exempt
def api_detect(request):
    """API endpoint for AI image detection (requires API key)"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    # Require API key authentication
    is_authenticated, error_response = require_api_key(request)
    if not is_authenticated:
        return error_response
    
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({
            'success': False,
            'error': 'Invalid form data',
            'errors': form.errors
        }, status=400)
    
    # Additional file validation
    image_file = form.cleaned_data.get('image')
    if image_file:
        is_valid, error_msg = validate_image_file(image_file)
        if not is_valid:
            return JsonResponse({
                'success': False,
                'error': error_msg
            }, status=400)
    
    image_upload = form.save()
    
    try:
        detection_service = get_detection_service()
        if detection_service is None:
            return JsonResponse({
                'success': False,
                'error': 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).'
            }, status=503)
        result = detection_service.detect_ai_image(image_upload.image.path)
        
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            }, status=500)

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
            'caption': result.get('caption', '')
        })
    except Exception as e:
        logger.error(f"API detection error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_POST
def submit_feedback(request, image_id):
    """Submit user feedback for an image detection result (CSRF protected)"""
    try:
        image_upload = get_object_or_404(ImageUpload, id=image_id)
        
        data = json.loads(request.body)
        feedback = data.get('feedback')
        notes = data.get('notes', '')
        
        if feedback not in ['correct', 'incorrect', 'unsure']:
            return JsonResponse({
                'success': False,
                'error': 'Invalid feedback value'
            })
        
        # Save feedback
        image_upload.save_feedback(feedback, notes)
        
        # Record feedback in learning system for re-upload detection
        try:
            from detector.feedback_learning import get_feedback_service
            feedback_service = get_feedback_service()
            
            # Determine correct answer from feedback
            correct_answer = None
            if feedback == 'incorrect':
                # If user says we're wrong, the correct answer is opposite of our prediction
                correct_answer = not image_upload.is_ai_generated
            elif feedback == 'correct':
                # If user confirms, our prediction was right
                correct_answer = image_upload.is_ai_generated
            
            # Record feedback with image hash
            if image_upload.image and hasattr(image_upload.image, 'path'):
                feedback_service.record_feedback(
                    image_hash=None,  # Will compute from path
                    prediction=image_upload.is_ai_generated,
                    confidence=image_upload.confidence_score,
                    user_feedback=feedback,
                    correct_answer=correct_answer
                )
                # Compute and save hash
                image_hash = feedback_service.compute_image_hash(image_upload.image.path)
                if image_hash:
                    feedback_service.record_feedback(
                        image_hash=image_hash,
                        prediction=image_upload.is_ai_generated,
                        confidence=image_upload.confidence_score,
                        user_feedback=feedback,
                        correct_answer=correct_answer
                    )
        except Exception as e:
            # Don't fail feedback submission if learning fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Feedback learning recording failed: {e}")
        
        # Trigger adaptive learning (asynchronously)
        try:
            from detector.adaptive_learning_service import adaptive_learning_service
            adaptive_learning_service.trigger_learning_on_feedback(image_upload)
        except Exception as e:
            # Don't fail feedback submission if learning fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Adaptive learning failed: {e}")
        
        return JsonResponse({
            'success': True,
            'message': 'Feedback submitted successfully',
            'feedback_status': image_upload.get_feedback_status()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def feedback_stats(request):
    """Display feedback statistics"""
    total_images = ImageUpload.objects.count()
    feedback_images = ImageUpload.objects.exclude(user_feedback='').count()
    
    correct_count = ImageUpload.objects.filter(user_feedback='correct').count()
    incorrect_count = ImageUpload.objects.filter(user_feedback='incorrect').count()
    unsure_count = ImageUpload.objects.filter(user_feedback='unsure').count()
    
    # Calculate accuracy based on feedback
    total_feedback = correct_count + incorrect_count
    accuracy = (correct_count / total_feedback * 100) if total_feedback > 0 else 0
    
    context = {
        'total_images': total_images,
        'feedback_images': feedback_images,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'unsure_count': unsure_count,
        'accuracy': accuracy,
        'feedback_rate': (feedback_images / total_images * 100) if total_images > 0 else 0
    }
    
    return render(request, 'detector/feedback_stats.html', context)

def batch_upload(request):
    """Batch upload page for multiple images - limited to 10 files"""
    if request.method == 'POST':
        files = request.FILES.getlist('images')
        results = []
        rejected_files = []
        
        # Limit to first 10 files
        max_files = 10
        files_to_process = files[:max_files]
        rejected_files_list = files[max_files:]
        
        # Process accepted files
        for file in files_to_process:
            try:
                # Validate file before processing
                is_valid, error_msg = validate_image_file(file)
                if not is_valid:
                    results.append({
                        'filename': file.name,
                        'error': error_msg,
                        'success': False
                    })
                    continue
                
                # Create ImageUpload instance
                image_upload = ImageUpload(image=file)
                image_upload.save()
                
                # Perform AI detection using comparative service
                detection_service = get_detection_service()
                if detection_service is None:
                    results.append({
                        'filename': file.name,
                        'error': 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).',
                        'success': False
                    })
                    continue
                result = detection_service.detect_ai_image(image_upload.image.path)
                
                if 'error' not in result:
                    image_upload.is_ai_generated = result['is_ai_generated']
                    image_upload.confidence_score = result['confidence']
                    image_upload.detection_indicators = result['indicators']
                    image_upload.analysis_details = result['analysis_details']
                    image_upload.method = result.get('method', 'unknown')
                    image_upload.save()
                    
                    results.append({
                        'id': image_upload.id,
                        'filename': image_upload.filename,
                        'is_ai_generated': result['is_ai_generated'],
                        'confidence': result['confidence'],
                        'success': True
                    })
                else:
                    results.append({
                        'filename': file.name,
                        'error': result['error'],
                        'success': False
                    })
                    
            except Exception as e:
                results.append({
                    'filename': file.name,
                    'error': str(e),
                    'success': False
                })
        
        # Add rejected files info
        for file in rejected_files_list:
            rejected_files.append({
                'filename': file.name,
                'reason': 'Maximum 10 files per batch. Only the first 10 files are processed.'
            })
        
        return JsonResponse({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'rejected_files': rejected_files,
            'message': f'Processed {len(files_to_process)} file(s)' + (f', rejected {len(rejected_files)} file(s) (max 10 allowed)' if rejected_files else '')
        })
    
    return render(request, 'detector/batch_upload.html')

def batch_results(request):
    """Display batch processing results"""
    # Get recent uploads for batch results
    recent_uploads = ImageUpload.objects.all().order_by('-uploaded_at')[:20]
    
    context = {
        'recent_uploads': recent_uploads
    }
    
    return render(request, 'detector/batch_results.html', context)

def individual_results(request):
    """Display all individual image detection results with pagination and filtering"""
    from django.core.paginator import Paginator
    from django.db.models import Q
    
    # Get filter and sort parameters
    filter_type = request.GET.get('filter', 'all')
    sort_by = request.GET.get('sort', '-uploaded_at')
    search_query = request.GET.get('search', '')
    
    # Base queryset
    queryset = ImageUpload.objects.all()
    
    # Apply filters
    if filter_type == 'real':
        queryset = queryset.filter(is_ai_generated=False)
    elif filter_type == 'ai':
        queryset = queryset.filter(is_ai_generated=True)
    elif filter_type == 'high_confidence':
        queryset = queryset.filter(confidence_score__gte=0.8)
    elif filter_type == 'low_confidence':
        queryset = queryset.filter(confidence_score__lt=0.5)
    
    # Apply search
    if search_query:
        queryset = queryset.filter(
            Q(image__icontains=search_query) |
            Q(method__icontains=search_query) |
            Q(detection_indicators__icontains=search_query)
        )
    
    # Apply sorting
    queryset = queryset.order_by(sort_by)
    
    # Pagination
    paginator = Paginator(queryset, 20)  # 20 images per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    total_images = ImageUpload.objects.count()
    real_images = ImageUpload.objects.filter(is_ai_generated=False).count()
    ai_images = ImageUpload.objects.filter(is_ai_generated=True).count()
    high_confidence = ImageUpload.objects.filter(confidence_score__gte=0.8).count()
    low_confidence = ImageUpload.objects.filter(confidence_score__lt=0.5).count()
    
    # Feedback statistics
    total_feedback = ImageUpload.objects.exclude(user_feedback='').count()
    correct_feedback = ImageUpload.objects.filter(user_feedback='correct').count()
    incorrect_feedback = ImageUpload.objects.filter(user_feedback='incorrect').count()
    
    context = {
        'page_obj': page_obj,
        'filter_type': filter_type,
        'sort_by': sort_by,
        'search_query': search_query,
        'total_images': total_images,
        'real_images': real_images,
        'ai_images': ai_images,
        'high_confidence': high_confidence,
        'low_confidence': low_confidence,
        'total_feedback': total_feedback,
        'correct_feedback': correct_feedback,
        'incorrect_feedback': incorrect_feedback,
        'feedback_rate': (total_feedback / total_images * 100) if total_images > 0 else 0,
    }
    
    return render(request, 'detector/individual_results.html', context)

def analytics_dashboard(request):
    """Comprehensive analytics dashboard"""
    from django.db.models import Count, Avg, Q
    from datetime import datetime, timedelta
    
    # Basic statistics
    total_images = ImageUpload.objects.count()
    ai_images = ImageUpload.objects.filter(is_ai_generated=True).count()
    real_images = ImageUpload.objects.filter(is_ai_generated=False).count()
    
    # Feedback statistics
    total_feedback = ImageUpload.objects.exclude(user_feedback='').count()
    correct_feedback = ImageUpload.objects.filter(user_feedback='correct').count()
    incorrect_feedback = ImageUpload.objects.filter(user_feedback='incorrect').count()
    unsure_feedback = ImageUpload.objects.filter(user_feedback='unsure').count()
    
    # Accuracy calculation
    total_decisions = correct_feedback + incorrect_feedback
    accuracy = (correct_feedback / total_decisions * 100) if total_decisions > 0 else 0
    
    # Confidence statistics
    avg_confidence = ImageUpload.objects.aggregate(avg_conf=Avg('confidence_score'))['avg_conf'] or 0
    high_confidence = ImageUpload.objects.filter(confidence_score__gte=0.8).count()
    medium_confidence = ImageUpload.objects.filter(confidence_score__gte=0.5, confidence_score__lt=0.8).count()
    low_confidence = ImageUpload.objects.filter(confidence_score__lt=0.5).count()
    
    # Method statistics
    method_stats = ImageUpload.objects.values('method').annotate(count=Count('method')).order_by('-count')
    
    # Recent activity (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)
    recent_uploads = ImageUpload.objects.filter(uploaded_at__gte=week_ago).count()
    recent_feedback = ImageUpload.objects.filter(feedback_timestamp__gte=week_ago).count()
    
    # Daily uploads for chart
    daily_uploads = []
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        count = ImageUpload.objects.filter(
            uploaded_at__date=date.date()
        ).count()
        daily_uploads.append({
            'date': date.strftime('%Y-%m-%d'),
            'count': count
        })
    
    # Confidence distribution
    confidence_ranges = [
        {'range': '0-20%', 'min': 0, 'max': 0.2, 'count': 0},
        {'range': '20-40%', 'min': 0.2, 'max': 0.4, 'count': 0},
        {'range': '40-60%', 'min': 0.4, 'max': 0.6, 'count': 0},
        {'range': '60-80%', 'min': 0.6, 'max': 0.8, 'count': 0},
        {'range': '80-100%', 'min': 0.8, 'max': 1.0, 'count': 0},
    ]
    
    for range_data in confidence_ranges:
        count = ImageUpload.objects.filter(
            confidence_score__gte=range_data['min'],
            confidence_score__lt=range_data['max']
        ).count()
        range_data['count'] = count
    
    context = {
        'total_images': total_images,
        'ai_images': ai_images,
        'real_images': real_images,
        'total_feedback': total_feedback,
        'correct_feedback': correct_feedback,
        'incorrect_feedback': incorrect_feedback,
        'unsure_feedback': unsure_feedback,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'avg_confidence_percentage': avg_confidence * 100,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'method_stats': method_stats,
        'recent_uploads': recent_uploads,
        'recent_feedback': recent_feedback,
        'daily_uploads': daily_uploads,
        'confidence_ranges': confidence_ranges,
        'feedback_rate': (total_feedback / total_images * 100) if total_images > 0 else 0
    }
    
    return render(request, 'detector/analytics_dashboard.html', context)

@csrf_exempt
def api_detect_realtime(request):
    """Real-time API endpoint for image detection (requires API key)"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    # Require API key authentication
    is_authenticated, error_response = require_api_key(request)
    if not is_authenticated:
        return error_response
    
    try:
        # Get image from request
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        
        # Comprehensive file validation
        is_valid, error_msg = validate_image_file(image_file)
        if not is_valid:
            return JsonResponse({'error': error_msg}, status=400)
        
        # Save temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_file.seek(0)
            tmp_file.write(image_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Perform detection using comparative service
            detection_service = get_detection_service()
            if detection_service is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).'
                }, status=503)
            result = detection_service.detect_ai_image(tmp_path)
            
            if 'error' in result:
                return JsonResponse({'error': result['error']}, status=500)
            
            # Return result
            return JsonResponse({
                'success': True,
                'is_ai_generated': result['is_ai_generated'],
                'confidence': result['confidence'],
                'method': result['method'],
                'indicators': result['indicators'],
                'analysis_details': result.get('analysis_details', {})
            })
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
def api_batch_detect(request):
    """API endpoint for batch image detection (requires API key)"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    # Require API key authentication
    is_authenticated, error_response = require_api_key(request)
    if not is_authenticated:
        return error_response
    
    try:
        files = request.FILES.getlist('images')
        
        if not files:
            return JsonResponse({'error': 'No images provided'}, status=400)
        
        if len(files) > 10:
            return JsonResponse({'error': 'Maximum 10 images per batch'}, status=400)
        
        results = []
        
        for file in files:
            try:
                # Comprehensive file validation
                is_valid, error_msg = validate_image_file(file)
                if not is_valid:
                    results.append({
                        'filename': file.name,
                        'success': False,
                        'error': error_msg
                    })
                    continue
                
                # Save temporary file with sanitized name
                import tempfile
                safe_filename = sanitize_filename(file.name)
                _, ext = os.path.splitext(safe_filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    file.seek(0)
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Perform detection using comparative service
                    detection_service = get_detection_service()
                    if detection_service is None:
                        results.append({
                            'filename': file.name,
                            'success': False,
                            'error': 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).'
                        })
                        continue
                    result = detection_service.detect_ai_image(tmp_path)
                    
                    if 'error' in result:
                        results.append({
                            'filename': file.name,
                            'success': False,
                            'error': result['error']
                        })
                    else:
                        results.append({
                            'filename': file.name,
                            'success': True,
                            'is_ai_generated': result['is_ai_generated'],
                            'confidence': result['confidence'],
                            'method': result['method'],
                            'indicators': result['indicators']
                        })
                    
                finally:
                    os.unlink(tmp_path)
                    
            except Exception as e:
                results.append({
                    'filename': file.name,
                    'success': False,
                    'error': str(e)
                })
        
        return JsonResponse({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def api_status(request):
    """API status endpoint"""
    try:
        detection_service = get_detection_service()
        if detection_service is None:
            return JsonResponse({
                'status': 'unavailable',
                'trained_model_available': False,
                'device': 'N/A',
                'message': 'Detection service is not available. PyTorch imports are disabled (ENABLE_MODEL_IMPORTS=0).',
                'timestamp': timezone.now().isoformat()
            })
        # Check if trained model is available
        has_trained_model = detection_service.trained_model is not None
        
        return JsonResponse({
            'status': 'operational',
            'trained_model_available': has_trained_model,
            'device': str(detection_service.device),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

def api_stats(request):
    """API statistics endpoint"""
    try:
        from django.db.models import Count, Avg
        
        # Get basic statistics
        total_images = ImageUpload.objects.count()
        ai_images = ImageUpload.objects.filter(is_ai_generated=True).count()
        real_images = ImageUpload.objects.filter(is_ai_generated=False).count()
        
        # Feedback statistics
        total_feedback = ImageUpload.objects.exclude(user_feedback='').count()
        correct_feedback = ImageUpload.objects.filter(user_feedback='correct').count()
        incorrect_feedback = ImageUpload.objects.filter(user_feedback='incorrect').count()
        
        # Accuracy calculation
        total_decisions = correct_feedback + incorrect_feedback
        accuracy = (correct_feedback / total_decisions * 100) if total_decisions > 0 else 0
        
        # Average confidence
        avg_confidence = ImageUpload.objects.aggregate(avg_conf=Avg('confidence_score'))['avg_conf'] or 0
        avg_confidence_percentage = avg_confidence * 100
        
        return JsonResponse({
            'total_images': total_images,
            'ai_images': ai_images,
            'real_images': real_images,
            'total_feedback': total_feedback,
            'correct_feedback': correct_feedback,
            'incorrect_feedback': incorrect_feedback,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_confidence_percentage': avg_confidence_percentage,
            'feedback_rate': (total_feedback / total_images * 100) if total_images > 0 else 0
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

def api_docs(request):
    """API documentation page"""
    return render(request, 'detector/api_docs.html')
