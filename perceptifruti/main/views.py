import os

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
from io import BytesIO
import base64


def home(request):
    return render(request, 'camera.html')


def receive_webcam_image(request):
    input_image_b64 = request.POST['image'].split(',')[1]
    image_pillow = Image.open(BytesIO(base64.b64decode(input_image_b64)))
    example_image = os.path.join(settings.STATICFILES_DIRS[0], 'img/example.png')
    image_extension = os.path.splitext(example_image)[1][1:]

    output_image_b64 = ''
    with open(example_image, 'rb') as f:
        output_image_b64 = base64.b64encode(f.read()).decode('utf-8')

    output_image_b64 = f'data:image/{image_extension};base64,{output_image_b64}'
    print(output_image_b64)

    return JsonResponse({'image': output_image_b64})
