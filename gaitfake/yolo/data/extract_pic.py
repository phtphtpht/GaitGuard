import os
import cv2

# 定义源文件夹A和目标文件夹C
src_folder = "/home/xk/pht/gaitfake/yolo/data/alpha1_motion8_merged"  # 替换为文件夹A的路径
dst_folder = "/home/xk/pht/gaitfake/yolo/data/alpha1_motion8_merged_pic"  # 替换为文件夹C的路径

# 创建目标文件夹C，如果不存在则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 遍历文件夹A下的所有子文件夹
for sub_folder in os.listdir(src_folder):
    sub_folder_path = os.path.join(src_folder, sub_folder)
    
    if os.path.isdir(sub_folder_path):  # 检查是否为文件夹
        # 在目标文件夹C下创建相应的子文件夹
        dst_sub_folder = os.path.join(dst_folder, sub_folder)
        if not os.path.exists(dst_sub_folder):
            os.makedirs(dst_sub_folder)
        
        # 遍历该子文件夹中的所有视频文件
        for video_file in os.listdir(sub_folder_path):
            video_file_path = os.path.join(sub_folder_path, video_file)
            
            # 创建每个视频对应的目标文件夹
            dst_video_folder = os.path.join(dst_sub_folder, os.path.splitext(video_file)[0])
            if not os.path.exists(dst_video_folder):
                os.makedirs(dst_video_folder)

            # 打开视频文件
            cap = cv2.VideoCapture(video_file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
            
            frame_num = 0  # 初始化帧计数器
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 提取第5, 10, 15等可被5整除的帧
                if frame_num % 5 == 0:
                    # 构造输出图片的路径和文件名
                    img_filename = f"frame_{frame_num}.jpg"
                    img_filepath = os.path.join(dst_video_folder, img_filename)

                    # 保存图片
                    cv2.imwrite(img_filepath, frame)

                frame_num += 1

            # 释放视频对象
            cap.release()

print("帧提取完成！")

id=[f'{i:03d}' for i in range(75,96)]
condition='nm-01'
view=['000','018','036','054','072','090','108','126','144','162','180']

