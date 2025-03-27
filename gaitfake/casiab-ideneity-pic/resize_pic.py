import os
import cv2
import numpy as np

#622*913
root_dir='/home/xk/pht/gaitfake/casiab-ideneity-pic/selected_raw_pic'
output_dir='/home/xk/pht/gaitfake/casiab-ideneity-pic/extracted_pic'
files= os.listdir(root_dir)
# 过滤出以 .png 结尾的文件
pics = [i for i in files if i.endswith('.png')]
for pic_ in pics:
    pic=cv2.imread(os.path.join(root_dir,pic_))
    # 获取图像的高度和宽度
    height, width, _ = pic.shape
    target_ratio = 913 / 622
    target_width = int(height / target_ratio)
    padding = (target_width - width) // 2
    background = np.full((height, target_width, 3), (85,128,105 ), dtype=np.uint8) #799061
    background[:, padding:padding+width] = pic
    background=cv2.resize(background,(622,913))
    cv2.imwrite(os.path.join(output_dir,pic_), background)