import os

# 设置文件所在目录路径
directory = "/home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/output/20241106/2218--loop/075"  # 替换为实际的目录路径

# 遍历目录下所有文件
for file_name in os.listdir(directory):
    if file_name.startswith("075-nm-01-000_") and file_name.endswith(".mp4"):
        # 分离出 alpha 和 beta 值
        parts = file_name.split("_")
        alpha_part = parts[1].replace("alph", "")
        beta_part = parts[2].replace("beta", "").replace(".mp4", "")
        
        # 计算新的 alpha 和 beta 值的编码
        alpha_code = f"{int(float(alpha_part) * 10):02}"
        beta_code = f"{int(float(beta_part) * 10):02}"
        
        # 创建新的文件名
        new_file_name = f"075-nm-01-{alpha_code}{beta_code}.mp4"
        
        # 执行重命名
        os.rename(os.path.join(directory, file_name), os.path.join(directory, new_file_name))
        print(f"Renamed '{file_name}' to '{new_file_name}'")

print("Renaming complete.")
