import cv2

# 读取AVI视频
input_path = '/home/xk/pht/gaitfake/yolo/data/casiab/001-nm-01-000.avi'
output_path = '000.mp4'

cap = cv2.VideoCapture(input_path)

# 获取视频帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置输出视频格式为MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 写入帧到输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
