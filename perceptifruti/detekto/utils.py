import os
from loguru import logger

import cv2
import torch


from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis
from .predictor import Predictor


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
CONF = 0.25
NMS = 0.1
TSIZE = 640


# def get_image_list(path):
#     image_names = []
#     for maindir, subdir, file_name_list in os.walk(path):
#         for filename in file_name_list:
#             apath = os.path.join(maindir, filename)
#             ext = os.path.splitext(apath)[1]
#             if ext in IMAGE_EXT:
#                 image_names.append(apath)
#     return image_names

# def image_demo(predictor, path):
#     if os.path.isdir(path):
#         files = get_image_list(path)
#     else:
#         files = [path]
#     files.sort()
    
#     image_name = files[0]
#     outputs, img_info = predictor.inference(image_name)
#     _, frameData = predictor.visual(outputs[0], img_info, predictor.confthre)
#     return img_info['raw_img'], frameData

# def show_bananas(img_info, frame_data):
#     for idx, data in frame_data.items():
#         x0, y0, x1, y1 = data['x0'], data['y0'], data['x1'], data['y1']
#         recorte = img_info[y0:y1, x0:x1]  # Recortar a imagem

#         # Salvar a imagem recortada
#         cv2.imwrite(f"out/detected_{idx}.jpg", recorte)

#         # Exibir a imagem recortada (opcional)
#         cv2.imshow(f"Recorte {idx}", recorte)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def get_banana_clusters(predictor, image_path):
    outputs, img_info = predictor.inference(image_path)
    banana_crops, frame_data = predictor.get_crops(outputs[0], img_info)

    return banana_crops, frame_data


def show_banana_crops(banana_crops):
    for idx, crop in enumerate(banana_crops):
        cv2.imshow(f"Cacho {idx+1}", crop)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    # img_info, frameData = image_demo(predictor, input_path)
    # show_bananas(img_info, frameData)
    banana_crops, frame_data = get_banana_clusters(predictor, input_path)
    if show_bananas:
        show_banana_crops(banana_crops)
    return banana_crops, frame_data


if __name__ == "__main__":
    INPUT_PATH = 'banana1.jpg'

    exp_name = 'yolox-s'
    exp = get_exp(None, exp_name)
    main(exp, INPUT_PATH, True)
