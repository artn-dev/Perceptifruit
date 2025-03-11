import json

from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
import base64


def home(request):
    return render(request, 'camera.html')


def receive_webcam_image(request):
    image_b64 = request.POST['image'].split(',')[1]
    image_pillow = Image.open(BytesIO(base64.b64decode(image_b64)))
    # Saves image
    # image_pillow.save('webcam.png')
    return HttpResponse(status=200)
