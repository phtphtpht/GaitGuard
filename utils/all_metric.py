import torch
import piqa
import cv2
from torchvision import transforms
from piqa import FID

# 初始化设备和指标
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr = piqa.PSNR().to(device)
ssim = piqa.SSIM().to(device)
lpips = piqa.LPIPS().to(device)
fid = FID().to(device)
ori_fid_tensor=torch.empty(0, 2048).to(device)
gen_fid_tensor=torch.empty(0, 2048).to(device)
# 图像预处理
transform = transforms.ToTensor()

# 计算视频帧间 PSNR、SSIM、LPIPS 和 FID
def calculate_frame_metrics(ori_frame, gen_frame):
    # 将 BGR 转换为 RGB
    ori_image_rgb = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    gen_image_rgb = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2RGB)
    
    # 将 numpy array 转换为 tensor
    ori_image_tensor = transform(ori_image_rgb).permute(0, 2, 1).unsqueeze(0).to(device)
    gen_image_tensor = transform(gen_image_rgb).permute(0, 2, 1).unsqueeze(0).to(device)

    # 计算每个指标
    psnr_score = psnr(ori_image_tensor, gen_image_tensor)
    ssim_score = ssim(ori_image_tensor, gen_image_tensor)
    lpips_score = lpips(ori_image_tensor, gen_image_tensor)
    ori_fid_tensor1=torch.cat((ori_fid_tensor,fid.features(ori_image_tensor)),dim=0)
    gen_fid_tensor1=torch.cat((gen_fid_tensor,fid.features(gen_image_tensor)),dim=0)
    fid_score = fid(ori_fid_tensor1, gen_fid_tensor1)
    
    return psnr_score.item(), ssim_score.item(), lpips_score.item(), fid_score.item()

# 计算生成视频的帧间指标
def calculate_video_metrics(video_path_gen):
    cap_gen = cv2.VideoCapture(video_path_gen)
    total_frames = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))
    psnr_scores, ssim_scores, lpips_scores,fid_scores = [], [], [],[]

    ret,prev_frame = cap_gen.read()
    for i in range(total_frames-2):
        ret, frame_gen = cap_gen.read()
        if not ret:
            print(f"Error reading frame {i} from generated video.")
            continue
        # print(prev_frame)
        psnr_score, ssim_score, lpips_score,fid_score = calculate_frame_metrics(prev_frame, frame_gen)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        fid_scores.append(fid_score)
        prev_frame=frame_gen

    # 计算所有帧间指标的平均值
    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    avg_fid = sum(fid_scores) / len(fid_scores) if fid_scores else 0

    return avg_psnr, avg_ssim, avg_lpips,avg_fid

# 示例：计算生成视频的帧间指标
gen_video_path = "/home/xk/pht/small_tools/perfect_vid_2/076/076-nm-01-"  # 替换为你的视频路径
num = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
gen_video_path2 = "/home/xk/pht/small_tools/perfect_vid_2/077/077-nm-01-"  # 替换为你的视频路径
video_path=[gen_video_path+i+'.mp4' for i in num]
video_path2=[gen_video_path2+i+'.mp4' for i in num]
avg_psnr, avg_ssim, avg_lpips=0,0,0
for i in video_path:
    avg_psnr1, avg_ssim1, avg_lpips1,avg_fid = calculate_video_metrics(i)
    avg_psnr+=avg_psnr1
    avg_ssim+=avg_ssim1
    avg_lpips+=avg_lpips1
for i in video_path2:
    avg_psnr1, avg_ssim1, avg_lpips1,avg_fid = calculate_video_metrics(i)
    avg_psnr+=avg_psnr1
    avg_ssim+=avg_ssim1
    avg_lpips+=avg_lpips1
# 输出平均指标
print(f"Average PSNR between consecutive frames: {avg_psnr/22}")
print(f"Average SSIM between consecutive frames: {avg_ssim/22}")
print(f"Average LPIPS between consecutive frames: {avg_lpips/22}")
# print(f"Average FID between consecutive frames: {avg_fid}")
