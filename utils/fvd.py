import torch
import numpy as np
import cv2
from torchvision import models, transforms
from scipy.linalg import sqrtm
from PIL import Image

# 选择设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的InceptionV3模型并将其转移到GPU（如果有）
inception_model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
inception_model.eval().to(device)  # 将模型加载到GPU

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 提取视频特征
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将frame从NumPy数组转换为PIL图像
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 对每一帧进行预处理
        frame = preprocess(frame_pil)
        frame = frame.unsqueeze(0).to(device)  # 转移到GPU
        
        with torch.no_grad():
            feature = inception_model(frame)
        features.append(feature.cpu().numpy().flatten())  # 转回CPU并提取特征
    
    cap.release()
    return np.array(features)

# 计算FVD
def calculate_fvd(real_video_path, generated_video_path):
    real_features = extract_video_features(real_video_path)
    generated_features = extract_video_features(generated_video_path)
    
    # 计算均值和协方差矩阵
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # 确保协方差矩阵的正定性
    eigvals_real, eigvecs_real = np.linalg.eigh(sigma_real)
    eigvals_generated, eigvecs_generated = np.linalg.eigh(sigma_generated)
    
    # 如果有负的特征值，将其设置为一个小的正数
    eigvals_real[eigvals_real < 0] = 1e-6
    eigvals_generated[eigvals_generated < 0] = 1e-6
    
    sigma_real = eigvecs_real.dot(np.diag(eigvals_real)).dot(eigvecs_real.T)
    sigma_generated = eigvecs_generated.dot(np.diag(eigvals_generated)).dot(eigvecs_generated.T)
    
    # 对协方差矩阵进行修正，添加一个小的正则化项
    sigma_real += 1e-6 * np.eye(sigma_real.shape[0])
    sigma_generated += 1e-6 * np.eye(sigma_generated.shape[0])
    
    # 计算Frechet距离
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    
    # 如果有复数部分，将其忽略
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fvd = np.linalg.norm(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fvd

# 示例用法
gen_video_path = "/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-"  # 替换为你的视频路径
real_video_path = "/home/xk/pht/gaitfake/yolo/data/casiab/076-nm-01-"  # 替换为你的视频路径
gen_video_path2 = "/home/xk/pht/small_tools/perfect_vid_2/077/077-nm-01-"  # 替换为你的视频路径
real_video_path2 = "/home/xk/pht/gaitfake/yolo/data/casiab/077-nm-01-"  # 替换为你的视频路径
num = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]

gen_video_path_list = [gen_video_path + i + ".mp4" for i in num]
real_video_path_list = [real_video_path + i + ".avi" for i in num]
gen_video_path_list2 = [gen_video_path2 + i + ".mp4" for i in num]
real_video_path_list2 = [real_video_path2 + i + ".avi" for i in num]

fvd_score = 0
for i in range(11):
    temp = calculate_fvd(real_video_path_list[i], gen_video_path_list[i])
    print(f"{i}:{temp}")
    fvd_score += temp

for i in range(11):
    temp = calculate_fvd(real_video_path_list2[i], gen_video_path_list2[i])
    print(f"{i}:{temp}")
    fvd_score += temp

print("FVD Score:", fvd_score / 11)