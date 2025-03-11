from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('receive-webcam-image/', views.receive_webcam_image, name='receive_webcam_image'),
]
