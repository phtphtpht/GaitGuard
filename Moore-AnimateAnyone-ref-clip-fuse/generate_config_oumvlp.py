import os

lines = [
r'pretrained_base_model_path: "./pretrained_weights/stable-diffusion-v1-5/"',
r'pretrained_vae_path: "./pretrained_weights/sd-vae-ft-mse"',
r'image_encoder_path: "./pretrained_weights/image_encoder"',
r'denoising_unet_path: "./pretrained_weights/denoising_unet.pth"',
r'reference_unet_path: "./pretrained_weights/reference_unet.pth"',
r'pose_guider_path: "./pretrained_weights/pose_guider.pth"',
r'motion_module_path: "./pretrained_weights/motion_module.pth"',
r'inference_config: "./configs/inference/inference_v2.yaml"',
r'weight_dtype: "fp16"',
r'test_cases:'
]

reference_dir='/home/xk/pht/gaitfake/casiab-ideneity-pic/extracted_pic'
skeleton_dir='/home/xk/gaitrecognition/pht_code/oumvlp_skeleton/'

# condition=['bg-01','cl-01','nm-01']
condition=['nm-01']
views=['000','015','030','045','060','075','090','180','195','210','225','240','255','270']

reference_file=os.listdir(reference_dir)

config_file=open('/home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate4.yaml', 'w')
for line in lines:
    config_file.write(line+'\n')

# out_line=' \"'+'/home/xk/pht/Moore-AnimateAnyone/configs/inference/ref_images/anyone-11.png\":\n'
# config_file.write(out_line)
count=1
for pic_id in range(75,125):
    
    out_line=' \"'+reference_dir+'/'+str(pic_id).zfill(3)+'.png'+'\":\n'
    config_file.write(out_line)
    if pic_id==124:
        outline='  - \"'+reference_dir+'/'+str(75).zfill(3)+'.png'+'\"\n'
    else:
        outline='  - \"'+reference_dir+'/'+str(pic_id+1).zfill(3)+'.png'+'\"\n'
    config_file.write(outline)

    for cond in condition:
        
        for view in views:
            vid_name2=view+'_00.mp4'
            skeletoon_path3=os.path.join(skeleton_dir,str(count*2).zfill(5),vid_name2)
            outline3='  - \"'+skeletoon_path3+'\"\n'
            config_file.write(outline3)
            if pic_id==100:
                vid_name=str(75).zfill(3)+'-'+cond+'-'+view+'.mp4'
            else:
                vid_name=str(pic_id+1).zfill(3)+'-'+cond+'-'+view+'.mp4'

            skeleton_path=os.path.join(skeleton_dir,str((count+1)*2).zfill(5),vid_name2)
            outline2='  - \"'+skeleton_path+'\"\n'
            config_file.write(outline2)
    count+=1