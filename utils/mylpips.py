import lpips
import torch
import cv2
from torchvision import transforms
from PIL import Image

# 初始化LPIPS模型
loss_fn = lpips.LPIPS(net='vgg')  # 你可以选择'vgg'、'alex'等预训练网络

# 图像预处理（与训练时使用的预处理相同）
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小以适应网络
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
])

# 读取视频中的帧
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

# 计算两帧之间的LPIPS
def calculate_lpips(image1, image2):
    # 转换为Tensor并进行预处理
    image1_tensor = preprocess(image1).unsqueeze(0)
    image2_tensor = preprocess(image2).unsqueeze(0)
    
    # 计算LPIPS值
    lpips_score = loss_fn(image1_tensor, image2_tensor)
    
    return lpips_score.item()

def calculate_video_lpips(video_path):
    frames = read_video_frames(video_path)
    lpips_scores = []

    # 计算连续帧之间的LPIPS
    for i in range(1, len(frames)):
        lpips_score = calculate_lpips(frames[i-1], frames[i])
        lpips_scores.append(lpips_score)

    # 计算所有帧间LPIPS的平均值
    average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    return average_lpips

# 示例：计算视频的平均帧间LPIPS
video_path = '/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-000.mp4'  # 输入视频路径
average_lpips = calculate_video_lpips(video_path)

# 输出平均LPIPS值
print(f"Average LPIPS between consecutive frames: {average_lpips}")
