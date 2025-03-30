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
from classifier.apps import ClassifierConfig
from classifier.enums import Ripeness
from classifier.models import FruitReading

from .models import Fruit


def home(request):
    return render(request, 'camera.html')


def dashboard(request):
    return render(request, 'dashboard.html')


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
        img = draw_bboxes_with_classification(input_path, frame_data, tags or [])
        cv2.imwrite(filename, img)

        return filename

    def get_url(self, filename):
        basename = os.path.basename(filename)
        url = f'{self.request.scheme}://{self.request.get_host()}{settings.MEDIA_URL}images/out/{basename}'
        return url

    def post(self, request):
        # input_path = self.receive_webcam_image()
        input_path = '/app/assets/romero/WhatsApp Image 2025-03-30 at 16.19.22.jpeg'
        banana, _ = Fruit.objects.get_or_create(name='banana')

        banana_crops, frame_data  = perform_detection(Exp(), input_path)

        classifier = ClassifierConfig.model
        tags = []
        counts = {k.label: 0 for k in Ripeness}
        if classifier:
            for crop in banana_crops:
                class_ = classifier.classify_image(image_array=crop)
                label = Ripeness(class_).label

                tags.append(label)
                counts[label] += 1
                FruitReading.objects.create(fruit=banana, reading=class_)

        output_path = self.save_proccessed_img(input_path, frame_data, tags)
        output_url = self.get_url(output_path)

        # os.remove(input_path)

        return JsonResponse({
            'image_url': output_url,
            'verde': counts[Ripeness.GREEN.label],
            'amadurecendo': counts[Ripeness.RIPENING.label],
            'maduras': counts[Ripeness.RIPE.label],
            'passadas': counts[Ripeness.OVERRIPE.label],
            'cachos_analisados': len(frame_data),
        })
