import os
from loguru import logger

import cv2
import torch


from detection_yolox.yolox.exp import get_exp
from detection_yolox.yolox.utils import get_model_info, postprocess, vis
from predictor import Predictor


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

def draw_bboxes_with_classification(img, frame_data, tags):
    # Isso daqui tá hardcoded por enquanto. Em algum momento eu devo voltar e melhorar essas informações
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(0, 255, 0)
    font_scale=0.5 
    thickness=2


    # Para essa função, eu to supondo que as tags vão vir no formato ['green', 'ripe, 'green', 'ripe', 'overripe']
    # Quando decidir como que o controlador vai enviar isso, eu posso ajustar aqui
    for idx, data in frame_data.items():
        x0, y0, x1, y1 = data['x0'], data['y0'], data['x1'], data['y1']
        
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

        text = tags[idx]
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

def main(exp, input_path):
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
    # img_info, frameData = image_demo(predictor, input_path)
    # show_bananas(img_info, frameData)
    banana_crops, frame_data = get_banana_clusters(predictor, input_path)
    show_banana_crops(banana_crops)


if __name__ == "__main__":
    INPUT_PATH = 'banana1.jpg'

    exp_name = 'yolox-s'
    exp = get_exp(None, exp_name)
    main(exp, INPUT_PATH)
