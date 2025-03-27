import os
import shutil

source_base = "/home/xk/pht/gaitfake/skeleton_video1"
target_base = "/home/xk/pht/gaitfake/skeleton_video2"

# 确保目标路径存在
os.makedirs(target_base, exist_ok=True)

# 遍历 075 到 124 文件夹
for i in range(75, 125):
    source_folder = os.path.join(source_base, f"{i:03d}")
    target_folder = os.path.join(target_base, f"{i:03d}")
    nm01_folder = os.path.join(source_folder, "nm-01")

    # 如果 nm-01 文件夹存在，复制其内容
    if os.path.exists(nm01_folder):
        os.makedirs(target_folder, exist_ok=True)
        shutil.copytree(nm01_folder, os.path.join(target_folder, "nm-01"), dirs_exist_ok=True)
