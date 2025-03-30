import os
from loguru import logger

import cv2
import torch


from yolox.exp import get_exp
from yolox.utils import get_model_info
from .predictor import Predictor


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
CONF = 0.25
NMS = 0.1
TSIZE = 640


def get_banana_clusters(predictor, image_path):
    outputs, img_info = predictor.inference(image_path)
    banana_crops, frame_data = predictor.get_crops(outputs[0], img_info)

    return banana_crops, frame_data


def show_banana_crops(banana_crops):
    for idx, crop in enumerate(banana_crops):
        cv2.imshow(f"Cacho {idx+1}", crop)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_bboxes_with_classification(image_path, frame_data, tags=None):
    # Isso daqui tá hardcoded por enquanto. Em algum momento eu devo voltar e melhorar essas informações
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(0, 255, 0)
    font_scale=0.5 
    thickness=2

    img = cv2.imread(image_path)

    # Para essa função, eu to supondo que as tags vão vir no formato ['green', 'ripe, 'green', 'ripe', 'overripe']
    # Quando decidir como que o controlador vai enviar isso, eu posso ajustar aqui
    for idx, data in frame_data.items():
        x0, y0, x1, y1 = data['x0'], data['y0'], data['x1'], data['y1']
        
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

        text = str(tags[idx]) if tags else data['class']
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x0 + (x1 - x0 - text_size[0]) // 2 
        text_y = y0 - 10  

        
        text_background_color = (0, 0, 0)  
        cv2.rectangle(
            img, 
            (text_x - 2, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + 2),
            text_background_color, 
            -1
        )
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

    return img


def main(exp, input_path, show_bananas=False):
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    exp.test_conf = CONF
    exp.nmsthre = NMS
    exp.test_size = (TSIZE, TSIZE)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # model.cuda()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
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
    banana_crops, frame_data = get_banana_clusters(predictor, input_path)
    return banana_crops, frame_data


if __name__ == "__main__":
    INPUT_PATH = '/app/assets/banana1.jpg'

    exp_name = 'yolox-s'
    exp = get_exp(None, exp_name)
    main(exp, INPUT_PATH, True)
