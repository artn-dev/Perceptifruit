import os
from loguru import logger

import cv2

import torch

from detection_yolox.yolox.exp import get_exp
from detection_yolox.yolox.utils import get_model_info
from detection_yolox.tools.demo import *


INPUT_PATH = 'assets/Cachos_de_banana_764ee2e24b.png'
CONF = 0.25
NMS = 0.1
TSIZE = 640


def image_demo(predictor, path):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    image_name = files[0]
    outputs, img_info = predictor.inference(image_name)
    _, frameData = predictor.visual(outputs[0], img_info, predictor.confthre)
    return img_info['raw_img'], frameData


def show_bananas(img_info, frame_data):
    for idx, data in frame_data.items():
        x0, y0, x1, y1 = data['x0'], data['y0'], data['x1'], data['y1']
        recorte = img_info[y0:y1, x0:x1]  # Recortar a imagem

        # Salvar a imagem recortada
        cv2.imwrite(f"out/detected_{idx}.jpg", recorte)

        # Exibir a imagem recortada (opcional)
        cv2.imshow(f"Recorte {idx}", recorte)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(exp):
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

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

    predictor = Predictor(model, exp,
                          trt_file=trt_file,
                          decoder=decoder,
                          device='gpu')
    img_info, frameData = image_demo(predictor, INPUT_PATH)
    show_bananas(img_info, frameData)


if __name__ == "__main__":
    exp_name = 'yolox-s'
    exp = get_exp(None, exp_name)
    main(exp)
