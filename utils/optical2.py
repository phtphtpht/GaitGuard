import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame):
    """
    计算两帧之间的光流（运动场）。
    """
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # 使用Farneback方法计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    return flow

def compute_motion_field_consistency(flow1, flow2):
    """
    计算连续两帧之间的运动场一致性。 
    比较光流场的差异。
    """
    # 计算两帧光流之间的差异（欧氏距离）
    diff = np.sqrt((flow1[..., 0] - flow2[..., 0])**2 + (flow1[..., 1] - flow2[..., 1])**2)
    
    # 计算差异的平均值和方差
    mean_diff = np.mean(diff)
    variance_diff = np.var(diff)
    
    return mean_diff, variance_diff

def main(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to read the video frame.")
        return
    
    # 初始化prev_flow为None，因为第一帧没有前一帧来计算光流
    prev_flow = None

    # 存储运动一致性度量
    consistency_scores = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        # 计算光流场
        flow = compute_optical_flow(prev_frame, next_frame)

        # 如果prev_flow存在，计算一致性
        if prev_flow is not None:
            # 计算连续帧之间的运动一致性
            mean_diff, variance_diff = compute_motion_field_consistency(prev_flow, flow)
            consistency_scores.append(mean_diff)
        
        # 更新上一帧和上一帧的光流
        prev_frame = next_frame
        prev_flow = flow
    
    # 计算所有帧之间一致性的平均值
    average_consistency = np.mean(consistency_scores)
    print(f"Average Motion Consistency: {average_consistency}")
    
    cap.release()


# 输入视频路径
gen_video_path = "/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-"  # 替换为你的视频路径
num = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]

gen_video_path_list = [gen_video_path + i + ".mp4" for i in num]
for i in gen_video_path_list:
    main(i)
