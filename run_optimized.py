import os
import torch
import cv2
import json
import time
from models.yolov8_model import YoloV8
from models.fasterrcnn_inference import FasterRCNN
from models.trocr_inference import TROCR
import traceback
from tqdm import tqdm
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
from models.create_colors import Colors

colors = Colors()

def dirchecks(file_path):
    if not os.path.exists(file_path):
        print(f"[INFO] {datetime.datetime.now()}: Cannot find this directory:\n{file_path}. Please check.\nExiting!!!!\n")
        sys.exit(1)
    else:
        print(f"[INFO] {datetime.datetime.now()}: Found this directory:\n{file_path}.\n")

def plot_results(img, boxes, class_names, scores, det_th, classes, ocr_model):
    for i in range(len(class_names)):
        if scores[i] >= det_th:
            x1, y1, x2, y2 = boxes[i]
            cname = class_names[i]

            if ocr_model is not None:
                cropped_image = img[y1:y2, x1:x2]
                ocr_text = ocr_model(cropped_image)
                cv2.putText(img, cname + f" : {ocr_text}", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors(classes.index(cname)), 2)
            else:
                cv2.putText(img, cname + f" {round(float(scores[i]), 2)}", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors(classes.index(cname)), 2)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors(classes.index(cname)), 2)
    
    return img

def process_image(image_path, out_path, ocr_model, model, det_th, classes):
    img = cv2.imread(image_path)
    boxes, class_names, scores = model(img)
    res_img = plot_results(img, boxes, class_names, scores, det_th, classes, ocr_model)
    out_image_path = f"{out_path}/{os.path.basename(image_path)[:-4]}.png"
    cv2.imwrite(out_image_path, res_img)

def img_inferencing(image_dir, out_path, ocr_model, model, det_th, custom_name, classes):
    out_path = f"{out_path}/img_out/{custom_name}"
    os.makedirs(out_path, exist_ok=True)
    image_paths = [os.path.join(image_dir, im_name) for im_name in os.listdir(image_dir)]

    with ThreadPoolExecutor() as executor:
        for image_path in tqdm(image_paths):
            executor.submit(process_image, image_path, out_path, ocr_model, model, det_th, classes)

def main(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirchecks(params["image_dir"])

    if params["use_model"] == "fasterrcnn":
        model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
        detection_thr = params["models"]["fasterrcnn"]["det_th"]
    elif params["use_model"] == "yolov8":
        model = YoloV8(model_weights=params["models"]["yolov8"]["model_weights"], classes=params["classes"], device=device, score_thresh=params["models"]["yolov8"]["det_th"], iou_thres=params["models"]["yolov8"]["iou_th"])
        detection_thr = params["models"]["yolov8"]["det_th"]

    if params["use_ocr_model"] == "trocr":
        ocr_model = TROCR(model_weights=params["ocr_models"]["trocr"]["model_weights"], MODEL_NAME=params["ocr_models"]["trocr"]["MODEL_NAME"])
    else:
        ocr_model = None

    start_time = time.time()
    img_inferencing(params["image_dir"], out_path=params["output_dir"], ocr_model=ocr_model, model=model, det_th=detection_thr, custom_name=params["custom_name"], classes=params["classes"])
    end_time = time.time()
    total_time = end_time - start_time

    print(f"[INFO] {datetime.datetime.now()}: Total time taken: {total_time:.2f} seconds")

if __name__ == '__main__':
    try:
        with open('./model_jsons/paramx.json', 'r') as f:
            params = json.load(f)

        main(params)
    except Exception as e:
        print(f"\n [ERROR] {datetime.datetime.now()} \n")
        traceback.print_exception(*sys.exc_info())
