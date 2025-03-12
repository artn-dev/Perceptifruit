import base64
import cv2
import os

from datetime import datetime
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View

from detekto.utils import main as perform_detection, draw_bboxes_with_classification
from detekto.detection_yolox.exps.default.yolox_s import Exp


def home(request):
    return render(request, 'camera.html')


class DetectBananas(View):
    def receive_webcam_image(self):
        image_b64 = self.request.POST['image'].split(',')[1]
        image_bytes = base64.b64decode(image_b64)

        now_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        image_name = f'img-{now_str}.png'

        container_path = os.path.join(settings.MEDIA_ROOT, 'images', 'in')
        os.makedirs(container_path, exist_ok=True)

        filename = os.path.join(container_path, image_name)
        with open(filename, 'wb') as f:
            f.write(image_bytes)

        return filename

    def save_proccessed_img(self, input_path, frame_data, tags=None):
        now_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        image_name = f'pred-{now_str}.png'

        container_path = os.path.join(settings.MEDIA_ROOT, 'images', 'out')
        os.makedirs(container_path, exist_ok=True)

        filename = os.path.join(container_path, image_name)
        img = draw_bboxes_with_classification(input_path, frame_data)
        cv2.imwrite(filename, img)

        return filename

    def get_url(self, filename):
        basename = os.path.basename(filename)
        url = f'{self.request.scheme}://{self.request.get_host()}{settings.MEDIA_URL}images/out/{basename}'
        return url

    async def post(self, request):
        input_path = self.receive_webcam_image()
        banana_crops, frame_data  = perform_detection(Exp(), input_path)
        # TODO: Passar imagens de bananas para classificador gerar lista de classificações
        output_path = self.save_proccessed_img(input_path, frame_data)
        output_url = self.get_url(output_path)

        return JsonResponse({ 'image_url': output_url })

