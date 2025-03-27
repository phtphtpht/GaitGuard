import os
import cv2

# 数据定义
ids = [f'{i:03d}' for i in range(75, 96)]
condition = 'nm-01'
views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

# 源文件夹A和目标文件夹B路径
src_folder = "/home/xk/pht/gaitfake/yolo/data/casiab"  # 文件夹A路径
dst_folder = "/home/xk/pht/gaitfake/yolo/data/casiab-pic"  # 文件夹B路径

# 创建目标文件夹B，如果不存在则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 遍历所有 id, condition, view 的组合，构造视频文件名
for id in ids:
    for view in views:
        video_filename = f"{id}-{condition}-{view}.avi"
        video_filepath = os.path.join(src_folder, video_filename)
        
        if os.path.exists(video_filepath):
            print(f"处理视频: {video_filename}")
            
            # 创建目标文件夹结构
            dst_id_folder = os.path.join(dst_folder, id)
            dst_video_folder = os.path.join(dst_id_folder, f"{id}-{condition}-{view}")
            if not os.path.exists(dst_video_folder):
                os.makedirs(dst_video_folder)

            # 打开视频文件
            cap = cv2.VideoCapture(video_filepath)
            frame_num = 0  # 初始化帧计数器

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 提取第5, 10, 15等可被5整除的帧
                if frame_num % 5 == 0:
                    img_filename = f"frame_{frame_num}.jpg"
                    img_filepath = os.path.join(dst_video_folder, img_filename)
                    
                    # 保存帧为图片
                    cv2.imwrite(img_filepath, frame)

                frame_num += 1

            # 释放视频对象
            cap.release()

        else:
            print(f"视频文件 {video_filename} 不存在")

print("所有视频处理完成！")
