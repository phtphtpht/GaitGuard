import cv2
import argparse
import os

def match(ori_mask_pos,gen_mask_pos,ori_image,gen_img,gen_mask):
    x1,y1,x2,y2=ori_mask_pos
    x11,y11,x12,y12=gen_mask_pos
    ori_height,ori_wid=abs(y1-y2),abs(x1-x2)
    target_height,target_wid=abs(y11-y12),abs(x11-x12)
    cropped_gen_img=gen_img[y11:y12,x11:x12]
    cropped_gen_mask=gen_mask[y11:y12,x11:x12]
    new_wid=int(target_wid/target_height*ori_height)
    cropped_gen_img=cv2.resize(cropped_gen_img,(new_wid,ori_height))
    cropped_gen_mask=cv2.resize(cropped_gen_mask,(new_wid,ori_height))

    center_x=(x1+x2)//2
    left_x,right_x=0,0
    if center_x-new_wid//2<0:
        left_x=0
        right_x=new_wid
    elif center_x-new_wid//2+new_wid>ori_image.shape[1]:
        left_x=ori_image.shape[1]-new_wid
        right_x=ori_image.shape[1]
    else:
        left_x=center_x-new_wid//2
        right_x=center_x-new_wid//2+new_wid

    cropped_ori_img=ori_image[y1:y2,left_x:right_x]
    cropped_ori_img[cropped_gen_mask>0]=cropped_gen_img[cropped_gen_mask>0]

    ori_image[y1:y2,left_x:right_x]=cropped_ori_img
    return ori_image


def main():
    ori_data_root='/home/xk/pht/gaitfake/yolo/data/casiab'
    ori_data_txt_root='/home/xk/pht/gaitfake/yolo/data/casiabtxt'
    gen_data_root='/home/xk/pht/gaitfake/perfect_gen_vid_2'
    gen_data_mask_root='/home/xk/pht/gaitfake/generated_gait_sih_perfect_2'
    gen_data_txt_root='/home/xk/pht/gaitfake/yolo/data/perfect_txt_2'
    out_root='/home/xk/pht/gaitfake/yolo/data/perfect_merged_2'
    os.makedirs(out_root,exist_ok=True)
    for id in range(76,78):
        if id==100:
            continue
        views=['000','018','036','054','072','090','108','126','144','162','180']
        out_folder=os.path.join(out_root,str(id).zfill(3))
        os.makedirs(out_folder,exist_ok=True)
        for view in views:
            print('working on',id,view)
            ori_vid_name=str(id).zfill(3)+'-nm-01-'+view+'.avi'
            ori_vid_path=os.path.join(ori_data_root,ori_vid_name)
            gen_vid_name=str(id).zfill(3)+'-nm-01-'+view+'.mp4'
            gen_vid_path=os.path.join(gen_data_root,str(id).zfill(3),gen_vid_name)
            out_vid_name=str(id).zfill(3)+'-nm-01-'+view+'.mp4'
            out_vid_path=os.path.join(out_folder,out_vid_name)

            ori_vid=cv2.VideoCapture(ori_vid_path)
            gen_vid=cv2.VideoCapture(gen_vid_path)
            # 获取视频的宽度
            width = int(ori_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 获取视频的高度
            height = int(ori_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

            ori_txt=open('/home/xk/pht/gaitfake/yolo/data/casiabtxt/'+str(id).zfill(3)+'/nm-01/'+str(id).zfill(3)+'-nm-01-'+view+'.txt','r')
            # gen_txt=open('/home/xk/pht/gaitfake/yolo/data/casiab_gen_txt/'+str(id).zfill(3)+'/nm-01/'+str(id).zfill(3)+'-nm-01-'+view+'.txt','r')
            gen_txt=open(gen_data_txt_root+'/'+str(id).zfill(3)+'/nm-01/'+str(id).zfill(3)+'-nm-01-'+view+'.txt','r')
            # gen_mask_path=os.path.join('/home/xk/pht/gaitfake/generated_gait_sih',str(id).zfill(3),'nm-01',view)
            gen_mask_path=os.path.join(gen_data_mask_root,str(id).zfill(3),'nm-01',view)
            count=0

            bkg_vid=cv2.VideoCapture('/home/xk/pht/gaitfake/yolo/data/extend_bkgrd/'+str(id).zfill(3)+'-bkgrd-'+view+'.mp4')
            bkg_pic=bkg_vid.read()[1]
            # print('/home/xk/pht/gaitfake/yolo/data/casiabtxt/'+str(id).zfill(3)+'/nm-01/'+str(id).zfill(3)+'-nm-01-'+view+'.txt')
            while True:
                line= ori_txt.readline()
                if not line:
                    break
                ret, frame = ori_vid.read()
                if not ret:
                    break
                if line=='empty\n' or line=='\n':
                    continue
                line= list(map(int,line.strip().split(',')))
                ret2,frame2=gen_vid.read()
                line2=gen_txt.readline()
                if not line2 or line2=='\n':
                    break
                while line2=='empty\n':
                    line2=gen_txt.readline()
                    print("here:",line2)
                    print(str(id).zfill(3)+'-nm-01-'+view+'-'+str(count).zfill(3)+'.png')
                # print(line2)
                line2=list(map(int,line2.strip().split(',')))
                gen_mask=cv2.imread(os.path.join(gen_mask_path,str(id).zfill(3)+'-nm-01-'+view+'-'+str(count).zfill(3)+'.png'),cv2.IMREAD_GRAYSCALE)
                print(gen_mask.shape)
                temp_pic=bkg_pic.copy()
                merge_pic=match(line,line2,temp_pic,frame2,gen_mask)
                out_video.write(merge_pic)
                count+=1
            out_video.release()
            ori_vid.release()
            gen_vid.release()


if __name__=="__main__":
    main()