import cv2

# 视频文件路径
video_path = '/home/xk/pht/gaitfake/output_video_162.mp4'

# 创建视频捕捉对象
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {frame_count}")

# 释放视频捕捉对象
cap.release()