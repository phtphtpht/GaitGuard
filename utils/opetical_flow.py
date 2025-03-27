import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame):
    """
    计算两帧之间的光流场
    """
    # 将帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算光流场
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def compute_optical_flow_error(flow1, flow2):
    """
    计算两个光流场之间的误差（L2 距离）
    """
    # 计算光流向量的差异
    diff = flow1 - flow2

    # 计算 L2 误差（欧几里得距离）
    error = np.sqrt(np.sum(diff**2, axis=-1))
    return np.mean(error)

def temporal_consistency(video_frames):
    """
    计算视频的时间一致性（基于光流误差）
    """
    total_error = 0
    num_frames = len(video_frames)

    # 计算每一对相邻帧的光流误差
    for t in range(num_frames - 1):
        # 计算光流场
        flow_t = compute_optical_flow(video_frames[t], video_frames[t + 1])

        # 如果是真实视频，可以计算与生成视频的光流误差
        # 这里假设 video_frames 是生成视频的帧
        # 如果是真实视频，需要提供真实视频的光流场
        # flow_real_t = compute_optical_flow(real_frames[t], real_frames[t + 1])
        # error = compute_optical_flow_error(flow_t, flow_real_t)

        # 这里仅计算生成视频的光流稳定性
        if t < num_frames - 2:
            flow_t1 = compute_optical_flow(video_frames[t + 1], video_frames[t + 2])
            error = compute_optical_flow_error(flow_t, flow_t1)
            total_error += error

    # 返回平均光流误差
    return total_error / (num_frames - 2)

# 示例：加载视频帧
def load_video_frames(video_path):
    """
    从视频文件中加载帧
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# 示例：计算时间一致性
video_path = "/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-"  # 替换为你的视频路径
num=["000.mp4","018.mp4","036.mp4","054.mp4","072.mp4","090.mp4","108.mp4","126.mp4","144.mp4","162.mp4","180.mp4"]
video_path_list=[video_path+ i for i in num]
print(video_path_list)
consistency_score=0
for path in video_path_list:
    video_frames = load_video_frames(path)
    consistency_score += temporal_consistency(video_frames)
print(f"Temporal Consistency (Optical Flow Error): {consistency_score}")