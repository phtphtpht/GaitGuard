from pathlib import Path
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import datetime

root_path='/home/xk/pht/gaitfake/yolo/data'
m = YOLO('yolov8n-seg.pt')

for vid in os.listdir(root_path+'/casiab'):
    vid_ele=vid.split('.')[0].split('-')
    if len(vid_ele)!=4:
        continue
    vid_id,vid_con,vid_view=vid_ele[0],vid_ele[1]+'-'+vid_ele[2],vid_ele[3]
    # print(vid_id)
    if int(vid_id)<75:
        continue
    if vid_con != 'nm-01':
        continue
    txt_path=root_path+'/casiabtxt/'+vid_id+'/'+vid_con
    os.makedirs(txt_path,exist_ok=True)
    vid_path=root_path+'/casiab'+'/'+vid
    res = m.predict(vid_path,max_det=1, save=False, imgsz=320, conf=0.5,classes=[0],retina_masks=True)
    file=open(os.path.join(txt_path,vid.split('.')[0]+'.txt'),'w')
    for (count,r) in enumerate(res):
        if not r:
            file.write('empty\n')
            continue
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        file.write(f"{x1},{y1},{x2},{y2}\n")
    file.close()

        
