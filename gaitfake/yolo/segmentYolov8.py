from pathlib import Path
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import datetime


import argparse
# video_path = "/home/user/wjc1/pht/data/kunkunshort.mp4"
parser = argparse.ArgumentParser()
parser.add_argument("--LAMA", action="store_true")
parser.add_argument("--FOREGROUND", action="store_true")
parser.add_argument("--BACKGROUND", action="store_true")
parser.add_argument("--CROP", action="store_true")
parser.add_argument("--VIDEO", action="store_true")
parser.add_argument("--video_path", type=str)
parser.add_argument("--save_txt", action="store_true")
args = parser.parse_args()
LAMA=args.LAMA
FOREGROUND=args.FOREGROUND
BACKGROUND=args.BACKGROUND
CROP=args.CROP
VIDEO=args.VIDEO
SAVETXT=args.save_txt

video_path=args.video_path
video_capture = cv2.VideoCapture(video_path)
# 获取视频帧数
fps=video_capture.get(cv2.CAP_PROP_FPS)
if True:
    m = YOLO('yolov8n-seg.pt')
    res = m.predict(video_path,max_det=1, save=True, imgsz=320, conf=0.5,classes=[0],retina_masks=True)
    filename = 'saved_data.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(res, file)
else:
    with open('saved_data.pkl', 'rb') as file:
        res = pickle.load(file)
# 创建保存目录
main_folder = "segment_results"
os.makedirs(main_folder, exist_ok=True)
current_time = datetime.datetime.now()
subfolder_name = current_time.strftime("%Y%m%d_%H%M%S")
subfolder_path=""
if FOREGROUND:
    subfolder_name="foreground_"+subfolder_name
elif BACKGROUND:
    subfolder_name="background_"+subfolder_name
elif LAMA:
    subfolder_name="lama_"+subfolder_name

if CROP:
    subfolder_name="crop_"+subfolder_name
# if not SAVETXT:
subfolder_path = os.path.join(main_folder, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)
if SAVETXT:
    subfolder_path_txt=os.path.join(main_folder,"seg_txt_"+current_time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(subfolder_path_txt, exist_ok=True)
    file=open(os.path.join(subfolder_path_txt, "segment.txt"),'w')
    file.write(str(fps)+'\n')

# iterate detection results 
for (count,r) in enumerate(res):
    if not r:
        continue
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem
    # iterate each object contour
    label = r.names[r.boxes.cls.tolist().pop()]

    b_mask = np.zeros(img.shape[:2], np.uint8)

    # Create contour mask 
    contour = r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    # Choose one:

    # OPTION-1: Isolate object with black background
    if FOREGROUND:
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img) 

    #Isolate object to a black hole
    if BACKGROUND:
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        tmp_mat = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        tmp_mat=cv2.bitwise_xor(tmp_mat,mask3ch)
        isolated=cv2.bitwise_and(tmp_mat,img)
    if LAMA:
        # tmp_mat = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        isolated = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        # isolated=cv2.bitwise_and(tmp_mat,mask3ch)

    # OPTION-2: Isolate object with transparent background (when saved as PNG)
    # isolated = np.dstack([img, b_mask])

    # OPTIONAL: detection crop (from either OPT1 or OPT2)
    if CROP:
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        isolated = img[y1:y2, x1:x2]
    if SAVETXT:
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        file.write(f"{count},{label},{x1},{y1},{x2},{y2}\n")
    
    
    if LAMA:
        image_path1 = os.path.join(subfolder_path, str(count)+'.png')
        cv2.imwrite(image_path1,img)
        image_path = os.path.join(subfolder_path, str(count)+'_mask.png')
        kernel_size = 5  # 可以调整大小
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 进行膨胀操作
        isolated = cv2.dilate(isolated, kernel, iterations=5)
        cv2.imwrite(image_path,isolated)
    else:
        image_path = os.path.join(subfolder_path, str(count)+'.png')
        cv2.imwrite(image_path,isolated)

if SAVETXT:
    file.close()

if VIDEO:
    save_dir = main_folder+"/segment_video_"+current_time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    output_video = os.path.join(save_dir, "output.mp4")
    output_txt=os.path.join(save_dir,"crop.txt")
    file_output=open(output_txt,'w')

    # fourcc = cv2.VideoWriter_fourcc(".mp4")
    image_files=os.listdir(subfolder_path)
    max_width = max([cv2.imread(os.path.join(subfolder_path , f)).shape[1] for f in image_files])
    max_height = max([cv2.imread(os.path.join(subfolder_path , f)).shape[0] for f in image_files])
    output_width = max_width
    output_height = max_height
    output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))

    def get_number_from_filename(filename):
        # 从文件名中提取数字部分
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return 0
    def sort_filenames(filename_list):
        # 排序文件名列表
        return sorted(filename_list, key=get_number_from_filename) 
        
    def get_number(filename):
    # 提取文件名中的数字部分
        name, ext = os.path.splitext(filename)
        number = ''.join(filter(str.isdigit, name))
        return int(number) if number.isdigit() else -1
    # 遍历图像文件列表，将每个图像按中心点进行扩展并放置到拼接视频帧的对应位置
    sorted_image_files = sorted(image_files, key=get_number)
    # sorted_image_files = sort_filenames(image_files)
    for image_file in sorted_image_files:
        image_path = os.path.join(subfolder_path , image_file)
        image = cv2.imread(image_path)
        if CROP:
            # 计算当前图像在拼接视频帧中的起始坐标
            start_x = (output_width - image.shape[1]) // 2
            start_y = (output_height - image.shape[0]) // 2

            # 创建一个与输出视频帧相同尺寸的画布
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)

            # 将当前图像放置在画布上的对应位置
            canvas[start_y:start_y + image.shape[0], start_x:start_x + image.shape[1]] = image

            out_str=str(start_x)+","+str(start_x+image.shape[1])+","+str(start_y)+","+str(start_y+image.shape[0])+'\n'
            file_output.write(out_str)
            # 将画布写入输出视频
            output.write(canvas)
        else:
            output.write(image)

    output.release()
    file_output.close()