import cv2
import numpy as np

def compute_frame_difference(frame1, frame2):
    """
    计算两帧之间的像素差异（L1 距离）
    """
    # 将帧转换为灰度图像
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算 L1 差异
    diff = np.abs(frame1_gray.astype(np.float32) - frame2_gray.astype(np.float32))
    return np.mean(diff)

def temporal_consistency(video_frames):
    """
    计算视频的时间一致性（基于帧间差异）
    """
    total_difference = 0
    num_frames = len(video_frames)

    # 计算每一对相邻帧的差异
    for t in range(num_frames - 1):
        diff = compute_frame_difference(video_frames[t], video_frames[t + 1])
        total_difference += diff

    # 返回平均帧间差异
    return total_difference / (num_frames - 1)

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
consistency_score=0
for path in video_path_list:
    video_frames = load_video_frames(path)
    consistency_score += temporal_consistency(video_frames)
print(f"Temporal Consistency (Frame Difference): {consistency_score/11}")