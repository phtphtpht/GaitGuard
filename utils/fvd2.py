import torch
import numpy as np
import cv2
from torchvision import models, transforms
from scipy.linalg import sqrtm
from PIL import Image

# 加载预训练的I3D模型
def load_i3d_model():
    i3d = models.video.r3d_18(pretrained=True)
    i3d.eval()  # 将模型设置为评估模式
    return i3d

# 定义视频帧预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),  # I3D通常使用112x112尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 提取视频特征
def extract_video_features(video_path, i3d_model):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
        frame_pil = Image.fromarray(frame_rgb)  # 转换为PIL图像
        frame_tensor = preprocess(frame_pil)  # 进行预处理
        frames.append(frame_tensor)

    cap.release()

    # 将视频帧堆叠为一个3D张量 (T, C, H, W)
    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # 改变维度顺序为 (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # 增加batch维度

    # 使用模型提取特征
    with torch.no_grad():
        features = i3d_model(video_tensor)  # 提取视频特征

    features = features.squeeze()  # 去掉batch维度，得到形状为(400,)的特征向量
    return features.cpu().numpy()  # 返回numpy数组

# 计算FVD
def calculate_fvd(real_video_path, generated_video_path, i3d_model):
    # 提取真实视频和生成视频的特征
    real_features = extract_video_features(real_video_path, i3d_model)
    generated_features = extract_video_features(generated_video_path, i3d_model)

    print("Shape of real_features:", real_features.shape)
    print("Shape of generated_features:", generated_features.shape)

    # 计算均值和协方差矩阵
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features.T)  # 转置后计算协方差矩阵
    sigma_generated = np.cov(generated_features.T)  # 转置后计算协方差矩阵

    # 计算Frechet距离
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))

    # 如果有复数部分，将其忽略
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.linalg.norm(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fvd

# 主程序示例
if __name__ == "__main__":
    # 加载I3D模型
    i3d_model = load_i3d_model()

    # 设置真实视频和生成视频路径
    real_video_path = "/home/xk/pht/gaitfake/yolo/data/casiab/076-nm-01-000.avi"
    generated_video_path = "/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-000.mp4"

    # 计算FVD
    fvd_score = calculate_fvd(real_video_path, generated_video_path, i3d_model)
    print("FVD Score:", fvd_score)
