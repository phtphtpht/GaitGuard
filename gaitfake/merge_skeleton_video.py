#这段代码是对casiab本身已有的骨骼数据进行视频合并的代码
import cv2
import os
from natsort import natsorted
from tqdm import tqdm

# 设置输入文件夹路径和输出视频文件名
root_dir = '/home/xk/pht/gaitfake/testout1/'
output_dir='/home/xk/pht/gaitfake/skeleton_video1'


# 遍历目录结构
# for 001,002,003... in dir(testout):
for index,identity in enumerate(os.listdir(root_dir)):
    print('%d now identity %s' %(index,identity))
    #identity_path=001,002,003...
    identity_path = os.path.join(root_dir, identity)
    #if 001,002,003... is directory
    if os.path.isdir(identity_path):
        # for bg-01,cl-01,nm-01 in 001,002,003...
        for condition1 in os.listdir(identity_path):
            condition1_path = os.path.join(identity_path, condition1)
            # if bg-01,cl-01,nm-01 is directory:
            if os.path.isdir(condition1_path):
                #for 000,018,036,054... in bg-01,cl-01,nm-01:
                for condition2 in os.listdir(condition1_path):
                    condition2_path = os.path.join(condition1_path, condition2)
                    #if 000,018,036,054... is directory:
                    if os.path.isdir(condition2_path):
                        # 处理 condition2 文件夹中的文件
                            # 获取文件夹中的所有文件
                        files = os.listdir(condition2_path)
                        sorted_files = natsorted(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
                        pic_ele=files[0].split('.')[0].split('-')
                        pic_id,pic_cond,pic_cond_id,pic_view=pic_ele[0],pic_ele[1],pic_ele[2],pic_ele[3]
                        
                        nested_dir = os.path.join(output_dir, pic_id, pic_cond+'-'+pic_cond_id, pic_view)
                        # 创建多级目录
                        os.makedirs(nested_dir, exist_ok=True)
                        # 按照文件名中的最后三个数字进行排序
                        

                        # 设置文件路径
                        output_video = os.path.join(nested_dir, pic_id+'-'+pic_cond+'-'+pic_cond_id+'-'+pic_view+'.mp4')
                        # 获取第一个图像的尺寸
                        img = cv2.imread(os.path.join(condition2_path, sorted_files[0]))
                        height, width, _ = img.shape

                        # 设置视频编解码器和输出视频对象
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

                        # 逐个将图像添加到视频中
                        for file in sorted_files:
                            img = cv2.imread(os.path.join(condition2_path, file))
                            video.write(img)

                        # 释放资源
                        cv2.destroyAllWindows()
                        video.release()









# 获取文件夹中的所有文件
# files = os.listdir(input_folder)

# pic_ele=files.split('.')[0].split('-')
# pic_id,pic_cond,pic_cond_id,pic_view=pic_ele[0],pic_ele[1]+'-'+pic_ele[2],pic_ele[3],pic_ele[4]

# # 按照文件名中的最后三个数字进行排序
# sorted_files = natsorted(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

# # 获取第一个图像的尺寸
# img = cv2.imread(os.path.join(input_folder, sorted_files[0]))
# height, width, _ = img.shape

# # 设置视频编解码器和输出视频对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# # 逐个将图像添加到视频中
# for file in sorted_files:
#     img = cv2.imread(os.path.join(input_folder, file))
#     video.write(img)

# # 释放资源
# cv2.destroyAllWindows()
# video.release()

# print("视频已创建：", output_video)