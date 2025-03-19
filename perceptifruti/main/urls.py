from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('receive-webcam-image/', views.DetectBananas.as_view(), name='receive_webcam_image'),
]
