import cv2
import os

# 图片文件路径的目录
input_dir = '/home/xk/pht/gaitfake/yolo/segment_results/crop_20240720_105609'
# 输出视频文件路径
output_video = 'output_video.avi'

# 获取所有图片文件名，按顺序排序
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

# 初始化最大尺寸
max_width = 0
max_height = 0

# 找到最大尺寸
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        continue
    height, width = image.shape[:2]
    if width > max_width:
        max_width = width
    if height > max_height:
        max_height = height

# 设置视频编码器 (XVID) 和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # 你可以根据需要设置帧率
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (max_width, max_height))

# 统一缩放图片到最大尺寸并写入视频文件
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        continue
    resized_image = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
    video_writer.write(resized_image)

# 释放视频写入器
video_writer.release()

print("Video has been successfully created.")
