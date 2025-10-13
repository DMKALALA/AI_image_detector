from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('result/<int:pk>/', views.result, name='result'),
    path('api/detect/', views.api_detect, name='api_detect'),
]
