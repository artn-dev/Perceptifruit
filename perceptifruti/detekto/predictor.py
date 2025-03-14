import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess
from .visualize import vis


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def get_crops(self, output, img_info, cls_conf=0.35):
            """
            Extrai os recortes da imagem original com base nos bounding boxes detectados.
            Retorna: lista de imagens recortadas (banana_crops) e o frame_data.
            """
            ratio = img_info["ratio"]
            img = img_info["raw_img"]
            if output is None:
                return [], {}

            output = output.cpu()
            bboxes = output[:, 0:4]
            bboxes /= ratio

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            banana_crops = []
            frame_data = {}
            count = 0

            for i in range(len(bboxes)):
                score = scores[i]
                if score < cls_conf:
                    continue

                x0 = int(bboxes[i][0])
                y0 = int(bboxes[i][1])
                x1 = int(bboxes[i][2])
                y1 = int(bboxes[i][3])

                # Garantir que os limites estão dentro da imagem
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(img.shape[1], x1)
                y1 = min(img.shape[0], y1)

                crop = img[y0:y1, x0:x1]
                banana_crops.append(crop)

                frame_data[count] = {
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'class': f'{self.cls_names[int(cls[i])]}:{score*100:.1f}%'
                }
                count += 1

            return banana_crops, frame_data

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, frameData = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, frameData