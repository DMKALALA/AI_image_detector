from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('result/<int:pk>/', views.result, name='result'),
    
    # API Endpoints
    path('api/detect/', views.api_detect, name='api_detect'),
    path('api/detect/realtime/', views.api_detect_realtime, name='api_detect_realtime'),
    path('api/detect/batch/', views.api_batch_detect, name='api_batch_detect'),
    path('api/status/', views.api_status, name='api_status'),
    path('api/stats/', views.api_stats, name='api_stats'),
    path('api/docs/', views.api_docs, name='api_docs'),
    
    # User Interface
    path('feedback/<int:image_id>/', views.submit_feedback, name='submit_feedback'),
    path('feedback-stats/', views.feedback_stats, name='feedback_stats'),
    path('batch-upload/', views.batch_upload, name='batch_upload'),
    path('batch-results/', views.batch_results, name='batch_results'),
    path('individual-results/', views.individual_results, name='individual_results'),
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
]
