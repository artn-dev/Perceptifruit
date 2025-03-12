from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64
import os

from detekto.utils import main as perform_detection, get_exp
from detekto.detection_yolox.exps.default.yolox_s import Exp


def home(request):
    return render(request, 'camera.html')


def receive_webcam_image(request):
    image_b64 = request.POST['image'].split(',')[1]
    image_pillow = Image.open(BytesIO(base64.b64decode(image_b64)))
    # Saves image
    # image_pillow.save('webcam.png')
    return HttpResponse(status=200)


class DetectBananas(View):
    def receive_webcam_image(self):
        image_b64 = self.request.POST['image'].split(',')[1]
        image_bytes = base64.b64decode(image_b64)

        now_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        image_name = f'img-{now_str}.png'
        container_path = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(container_path, exist_ok=True)

        fullpath = os.path.join(container_path, image_name)
        with open(fullpath, 'wb') as f:
            f.write(image_bytes)

        return fullpath

    async def post(self, request):
        input_path = self.receive_webcam_image()

        exp = Exp()
        banana_crops, _  = perform_detection(exp, input_path)

        return HttpResponse(status=200)

