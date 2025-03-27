import os
import cv2
root_dir='/home/xk/pht/gaitfake/casiab-ideneity-pic'

for folder in os.listdir(root_dir):
    folder_path=os.path.join(root_dir,folder)
    if os.path.isdir(folder_path):
        pic_id=folder_path.split('/')[-1]
        print(folder_path)
        pic= os.listdir(folder_path)
        img=cv2.imread(os.path.join(folder_path,pic[0]))
        cv2.imwrite(os.path.join(root_dir,pic_id+'.png'),img)
        