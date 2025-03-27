from ultralytics import YOLO
import cv2
import os
import argparse
import numpy as np
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--img_path",type=str)
parser.add_argument("--FOREGROUND_BLACK", action="store_true")
parser.add_argument("--FOREGROUND_WHITE", action="store_true")
args=parser.parse_args()
FOREGROUND_BLACK=args.FOREGROUND_BLACK
FOREGROUND_WHITE=args.FOREGROUND_WHITE
img_path=args.img_path
m = YOLO('yolov8n-seg.pt')
res = m.predict(img_path,max_det=1, save=True, imgsz=320, conf=0.5,classes=[0],retina_masks=True)

for r in res:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem
    # iterate each object contour
    label = r.names[r.boxes.cls.tolist().pop()]

    b_mask = np.zeros(img.shape[:2], np.uint8)

    # Create contour mask 
    contour = r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        # OPTION-1: Isolate object with black background
    if FOREGROUND_BLACK:
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img) 

    #White background
    if FOREGROUND_WHITE:
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img) 
        isolated[np.where((mask3ch==[0,0,0]).all(axis=2))] = [255,255,255]

    img_path=os.getcwd()+'/'+img_name+".png"
    cv2.imwrite(img_path,isolated)