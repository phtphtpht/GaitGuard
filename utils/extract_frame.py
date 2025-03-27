import cv2
import os

def extract_frames(video_folder, output_base_folder, frame_interval=5):
    # 确保输出基础文件夹存在
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        # 获取视频文件名(不含扩展名)作为输出子文件夹名
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(output_base_folder, video_name)
        
        # 为每个视频创建对应的输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 打开视频
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        saved_count = 0
        
        print(f"正在处理视频: {video_file}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔frame_interval帧保存一次
            if frame_count % frame_interval == 0:
                # 生成输出文件名 例如: frame_0000.jpg
                output_path = os.path.join(output_folder, f"frame_{saved_count:04d}_any13.png")
                cv2.imwrite(output_path, frame)
                saved_count += 1
            
            frame_count += 1
            
        cap.release()
        print(f"视频 {video_file} 处理完成，共保存 {saved_count} 帧")
    
    print("所有视频处理完成!")

# 使用示例
video_folder = "/home/xk/pht/dataiccv/hzc"  # 包含视频的文件夹路径
output_base_folder = "/home/xk/pht/dataiccv/hzc/frames"  # 存储提取帧的基础文件夹路径
frame_interval = 5  # 每5帧提取一帧

extract_frames(video_folder, output_base_folder, frame_interval)