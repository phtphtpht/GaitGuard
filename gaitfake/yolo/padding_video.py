import cv2

# 设置目标高宽比
target_ratio = 512/784

# 读取视频
cap = cv2.VideoCapture('/home/xk/pht/gaitfake/yolo/output_video_kps.mp4')

# 获取视频帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算当前高宽比
current_ratio = width / height

# 确定新的宽度和高度
if current_ratio > target_ratio:
    new_width = width
    new_height = int(width / target_ratio)
else:
    new_height = height
    new_width = int(height * target_ratio)

# 输出视频设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('padding_kps.mp4', fourcc, fps, (new_width, new_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 计算填充的尺寸
    top = (new_height - height) // 2
    bottom = new_height - height - top
    left = (new_width - width) // 2
    right = new_width - width - left

    # 填充
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 写入帧到输出视频
    out.write(padded_frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
