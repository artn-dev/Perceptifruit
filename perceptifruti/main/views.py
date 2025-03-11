from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from PIL import Image
from io import BytesIO
import base64

from detekto.utils import main as perform_detection, get_exp


def home(request):
    return render(request, 'camera.html')


def receive_webcam_image(request):
    image_b64 = request.POST['image'].split(',')[1]
    image_pillow = Image.open(BytesIO(base64.b64decode(image_b64)))
    # Saves image
    # image_pillow.save('webcam.png')
    return HttpResponse(status=200)


class Foo(View):
    def receive_webcam_image(self):
        image = self.request.FILES['image']

        fs = FileSystemStorage(location='media/images/')
        image_name = fs.save(image.name, image)

        return fs.path(image_name)

    async def post(self, request):
        input_path = self.receive_webcam_image()

        exp = get_exp(None, 'yolox-s')
        perform_detection(exp, input_path)
