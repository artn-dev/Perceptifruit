import argparse
import os
import time
from loguru import logger

import cv2
import json

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from tools.demo import *


INPUT_PATH = 'assets/banana.png'
CONF = 0.25
NMS = 0.45
TSIZE = 640


def show_bananas(path, img_info, frame_data):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    image_name = files[0]

    for idx, data in frame_data.items():
        x0, y0, x1, y1 = data['x0'], data['y0'], data['x1'], data['y1']
        recorte = img_info[y0:y1, x0:x1]  # Recortar a imagem

        # Salvar a imagem recortada
        cv2.imwrite(f"detected_{idx}.jpg", recorte)

        # Exibir a imagem recortada (opcional)
        cv2.imshow(f"Recorte {idx}", recorte)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(exp):
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
        
    exp.test_conf = CONF
    exp.nmsthre = NMS
    exp.test_size = (TSIZE, TSIZE)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.cuda()
    model.eval()

    ckpt_file = 'yolox-dist/yolox_s.pth'
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        'gpu', False, False,
    )
    current_time = time.localtime()
    img_info, frameData = image_demo(predictor, vis_folder, INPUT_PATH, current_time, False)
    show_bananas(INPUT_PATH, img_info, frameData)


if __name__ == "__main__":
    exp_name = 'yolox-s'
    exp = get_exp(None, exp_name)
    main(exp)
