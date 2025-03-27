#这个文件可以对每个身份的人物从视频中进行crop提取
from pathlib import Path
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import datetime
import argparse

root_dir='/home/xk/pht/gaitfake/yolo/data/casiab'
identity=[str(i).zfill(3) for i in range(1,124)]
wanted=[124]
wanted=[str(i).zfill(3) for i in wanted]
target_video=[i+'-nm-01-000.avi' for i in wanted]
output_dir='/home/xk/pht/gaitfake/casiab-ideneity-pic-2'



for num,video in enumerate(target_video):
    video_path=os.path.join(root_dir,video)
    video_capture=cv2.VideoCapture(video_path)
    m = YOLO('yolov8n-seg.pt')
    res = m.predict(video_path,max_det=1, save=False, imgsz=320, conf=0.5,classes=[0],retina_masks=True)
    pic_id,pic_cond,pic_cond_id,pic_view=video.split('.')[0].split('-')
    output_folder=os.path.join(output_dir,pic_id)
    os.makedirs(output_folder, exist_ok=True)
    for count,r in enumerate(res[-40:-1]):
        if not r:
            continue
        img=np.copy(r.orig_img)
        label=r.names[r.boxes.cls.tolist().pop()]
        b_mask=np.zeros(img.shape[:2],np.uint8)

        contour=r.masks.xy.pop().astype(np.int32).reshape(-1,1,2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        isolated = img[y1:y2, x1:x2]
        image_path=os.path.join(output_folder,str(count)+'.png')
        cv2.imwrite(image_path,isolated)
