#这个代码用来将步态视频中crop出来大小不一的人物图像缩放到同样大小
import cv2
import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
# 图片所在目录
# input_dir = '/home/xk/pht/gaitfake/yolo/segment_results/crop_20240722_105045'
# 保存调整后的图片的目录
# output_dir = '/home/xk/pht/gaitfake/yolo/resize_results/2'
args = parser.parse_args()
input_dir=args.input_dir
output_dir=args.output_dir
# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有图片文件名
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# 初始化最大尺寸
max_width = 0
max_height = 0

# 读取每张图片并确定最大尺寸
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        max_width = max(max_width, width)
        max_height = max(max_height, height)

# 调整所有图片的尺寸到最大尺寸
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is not None:
        # 缩放图片到最大尺寸
        resized_image = cv2.resize(image, (max_width, max_height))

        # 保存调整后的图片
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, resized_image)

print("All images have been resized to the maximum dimensions and saved.")
