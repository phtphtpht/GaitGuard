from pathlib import Path
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import datetime

import argparse

m = YOLO('yolov8n-seg.pt')

root_dir='/home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/output/20241109/2259--oumvlp_1'
output_dir='/home/xk/pht/gaitfake/oumvlp_2'
vid_id=[str(i).zfill(3) for i in range(75,125)]
# vid_id.append([str(i).zfill(3) for i in range(101,113)])

for index,identity in enumerate(os.listdir(root_dir)):
    identity_path=os.path.join(root_dir,identity)
    id_out_path=output_dir+'/'+identity+'/nm-01'

    os.makedirs(id_out_path,exist_ok=True)
    if os.path.isdir(identity_path):
        for vid in os.listdir(identity_path):
            vid_ori_path=os.path.join(identity_path,vid)
            vid_view=vid.split('.')[0].split('-')[-1]
            vid_out_path=id_out_path+'/'+vid_view
            os.makedirs(vid_out_path,exist_ok=True)
            #分割
            res = m.predict(vid_ori_path,max_det=1, save=False, imgsz=320, conf=0.5,classes=[0],retina_masks=True)

            for (count,r) in enumerate(res):
                if not r:
                    continue
                img = np.copy(r.orig_img)
                img_name = Path(r.path).stem
                # iterate each object contour
                label = r.names[r.boxes.cls.tolist().pop()]

                b_mask = np.zeros(img.shape[:2], np.uint8)
                contour = r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                image_path=vid_out_path+'/'+identity+'-nm-01-'+vid_view+'-'+str(count).zfill(3)+'.png'
                cv2.imwrite(image_path,mask3ch)



