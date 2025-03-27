import cv2

# # 打开视频文件
# video = cv2.VideoCapture('/home/xk/pht/gaitfake/yolo/data/perfect_merged/075/075-nm-01-000.mp4')

# # 获取总帧数
# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# print(f"视频总帧数: {total_frames}")

# # 记得释放视频对象
# video.release()

import numpy as np

def extend_video_last_frame(input_path, output_path, num_total=10):
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_copies=num_total-total_frames
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 首先写入原始视频的所有帧
    frame_count = 0
    last_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        out.write(frame)
        last_frame = frame.copy()
        frame_count += 1
    
    # 复制最后一帧指定次数
    for _ in range(num_copies):
        out.write(last_frame)
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"原始视频帧数: {frame_count}")
    print(f"扩充后视频帧数: {frame_count + num_copies}")


ori_data_root='/home/xk/pht/gaitfake/yolo/data/casiab'
views=['000','018','036','054','072','090','108','126','144','162','180']
out_root='/home/xk/pht/gaitfake/yolo/data/extend_bkgrd'
for view in views:
    video = cv2.VideoCapture(ori_data_root+'/077-nm-01-'+view+'.avi')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    extend_video_last_frame(ori_data_root+'/077-bkgrd-'+view+'.avi',out_root+'/077-bkgrd-'+view+'.mp4',num_total=total_frames)

# # 使用示例
# input_video = "input.mp4"
# output_video = "output.mp4"
# extend_video_last_frame(input_video, output_video, num_copies=10)