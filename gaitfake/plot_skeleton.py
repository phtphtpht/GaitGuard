#这段代码可以对骨骼数据进行绘制，骨骼重排到

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
import os
import pandas as pd
import csv
from tqdm import tqdm

'''
"keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
"skeleton": [
    [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
    [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
'''
N=8

'''
left_ankle <---> left_knee
left_knee <---> left_hip
left_hip <---> left_shoulder
right_hip <---> right_knee
right_knee <---> right_ankle
right_hip <---> right_shoulder
left_ear <---> left_eye
left_eye <---> right_eye
left_eye <---> nose
nose <---> right_eye
right_eye <---> right_ear
left_wrist <---> left_elbow
left_elbow <---> left_shoulder
left_shoulder <---> right_shoulder
right_shoulder <---> right_elbow
right_elbow <---> right_wrist
'''




class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            if type(self.color[i]) is tuple:
                self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))
            else:
                self.link_pairs[i].append(self.color[i])

        self.ring_color = []
        for i in range(len(self.point_color)):
            if type(self.point_color[i]) is tuple:
                self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
            else:
                self.ring_color.append(self.point_color[i])
            # self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
        
# Xiaochu Style
# (R,G,B)
color1 = [(179,0,0),(228,26,28),(255,255,51),
    (49,163,84), (0,109,45), (255,255,51),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (217,95,14), (254,153,41),(255,255,51),
    (44,127,184),(0,0,255)]

link_pairs1 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6],
        [6, 8], [8, 10],
        ]

point_color1 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,51),(255,255,51),
            (254,153,41),(44,127,184),
            (217,95,14),(0,0,255),
            (255,255,51),(255,255,51),(228,26,28),
            (49,163,84),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1)

fig_wid=320
fig_height=240
target_wid=512
target_height=784

# Chunhua Style
# (R,G,B)
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        ]

point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

# color3=['#0034A0','#00609C','#00899E','#008331','#008865','#007F00',
# (240,2,127),(240,2,127),(240,2,127),(240,2,127),(240,2,127)
# ,'#2C8400','#638B00',(169, 209, 142),'#996706','#989000']

color3=['#003399','#006699','#009999','#009933','#009966','#009900','#990066','#000099','#990099','#330099','#660099','#339900','#669900','#993300','#996600','#999900']

# point_color3=[(240,2,127),(240,2,127),(240,2,127),(240,2,127),(240,2,127),'#44E000','#FFA900','#00D800','#FFF20E','#00DC54','#A7EA00','#0056FF','#00DFB1','#0013FF','#00E5FF','#5E18FF','#009EFF']
point_color3=['#FF0000','#FF00FF','#AA00FF','#FF0055','#FF00AA','#55FF00','#FFAA00','#00FF00','#FFFF00','#00FF55','#AAFF00','#0055FF','#00FFAA','#0000FF','#00FFFF','#5500FF','#00AAFF']

keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
pht_style = ColorStyle(color3, link_pairs2, point_color3)

def plot_a_pic(csv_output_filename,keypoints_list,face_front):
    output_dir = os.path.dirname(csv_output_filename)
    # 如果目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csvfile = open(csv_output_filename, 'w', newline='')
    csvwriter=csv.writer(csvfile)
    csvwriter.writerow(csv_headers)
    csv_output_rows=[]
    for num,item_keypoints in enumerate(keypoints_list):
        fig, ax = plt.subplots(figsize=(target_wid / 100, target_height / 100), dpi=100)
        fig.patch.set_facecolor('black')  # 设置图形背景色
        ax.set_facecolor('black')
        ax.axis('off')
        ax.invert_yaxis()
        new_csv_row=[]
        for k, link_pair in enumerate(pht_style.link_pairs):
            x_values = [item_keypoints[link_pair[0]][0], item_keypoints[link_pair[1]][0]]
            y_values = [item_keypoints[link_pair[0]][1], item_keypoints[link_pair[1]][1]]
            plt.plot(x_values, y_values, ls='-', lw=4, alpha=1, color=pht_style.color[k], zorder=0)
        radius=min(target_height,target_wid)/100
        x_left_ear=item_keypoints[3][0]
        x_right_ear=item_keypoints[4][0]
        y_ear=(item_keypoints[3][1]+item_keypoints[4][1])/2

        left_eye=item_keypoints[1]
        right_eye=item_keypoints[2]
        nose=item_keypoints[0]
        n=8
        face_wid=abs(x_left_ear-x_right_ear)*(n-1)/n
        # face_height=60/21*17
        face_height=face_wid/21*17
        y_chin=y_ear+face_height
        x_chin=(x_left_ear+x_right_ear)/2

        nose=item_keypoints[0]
        
        for k,key_point in enumerate(item_keypoints):
            circle = mpatches.Circle(tuple(key_point), 
                                                        radius=radius, 
                                                        ec='black', 
                                                        fc=pht_style.ring_color[k], 
                                                        alpha=1, 
                                                        linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)
            new_csv_row.extend(key_point)
        if x_left_ear!=x_right_ear:
            a=(y_ear-y_chin)/(x_right_ear-(x_left_ear+x_right_ear)/2)**2
            # a=(y_ear-y_chin)/(x_left_ear**2-2*x_left_ear*x_chin+x_chin**2)
            # b=-2*a*x_chin
            # c=y_chin+a*x_chin**2
        if face_front:
            for i in range(0,18):
                x_plot=x_left_ear-face_wid/16*(i-1)
                # if i<=9:
                if x_left_ear!=x_right_ear:
                    y_plot=a*(x_plot-(x_left_ear+x_right_ear)/2)**2+y_chin
                    # y_plot=a*x_plot**2+b*x_plot+c

                else:
                    if i<=9:
                        y_plot=y_chin-(9-i)*face_height/9
                    else:
                        y_plot=y_chin-(i-9)*face_height/9
                # else:
                #     y_plot=y_ear+face_height/8*(17-i)
                circle = mpatches.Circle((x_plot,y_plot), 
                                        radius=radius, 
                                        ec='black', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle.set_zorder(1)
                ax.add_patch(circle)

            for i in range(5):
                x_plot_1=left_eye[0]-(i-2)*2
                y_plot_1=left_eye[1]
                x_plot_2=right_eye[0]-(i-2)*2
                y_plot_2=right_eye[1]
                circle1 = mpatches.Circle((x_plot_1,y_plot_1), 
                                        radius=radius, 
                                        ec='white', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle1.set_zorder(1)
                ax.add_patch(circle1)
                circle2 = mpatches.Circle((x_plot_2,y_plot_2), 
                                        radius=radius, 
                                        ec='white', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle2.set_zorder(1)
                ax.add_patch(circle2)
            for i in range(5):
                x_plot=nose[0]
                y_plot=nose[1]-(i-2)*2
                circle1 = mpatches.Circle((x_plot,y_plot), 
                                        radius=radius, 
                                        ec='white', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle1.set_zorder(1)
                ax.add_patch(circle1)
            for i in range(5):
                x_plot=nose[0]-(i-2)*1
                y_plot=nose[1]+2*2
                circle1 = mpatches.Circle((x_plot,y_plot), 
                                        radius=radius, 
                                        ec='white', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle1.set_zorder(1)
                ax.add_patch(circle1)
        
            for i in range(7):
                x_plot=nose[0]-(i-3)*2
                y_plot=nose[1]+4*2
                circle1 = mpatches.Circle((x_plot,y_plot), 
                                        radius=radius, 
                                        ec='white', 
                                        fc='white', 
                                        alpha=1, 
                                        linewidth=1)
                circle1.set_zorder(1)
                ax.add_patch(circle1)

        new_csv_row.extend(resize_scale_list[num])
        csv_output_rows.append(new_csv_row)
        plt.xlim(0,target_wid)
        plt.ylim(target_height,0)
        # 保存图像
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        path_name='./testout2/'+pre_pic_id+'/'+pre_pic_cond+'-'+pre_pic_cond_id+'/'+pre_pic_view
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        out_pic_name=pre_pic_id+'-'+pre_pic_cond+'-'+pre_pic_cond_id+'-'+pre_pic_view+'-'+str(num).zfill(3)+'.png'
        plt.savefig(path_name+'/'+out_pic_name)
        plt.close()
    csvwriter.writerows(csv_output_rows)
    csvfile.close()

if __name__=="__main__":
    data=pd.read_csv("/home/xk/gaitrecognition/gaitgraph/GaitGraph-main/GaitGraph-main/data/casia-b_pose_test.csv")
    # 创建一个空的列表来存储结果
    results = []
    pre_pic_id=""
    pre_pic_cond=""
    pre_pic_cond_id=""
    pre_pic_view=""
    keypoints_list=list()
    resize_scale_list=list()
    csv_output_rows=[]
    
    csv_headers=list()
    for i in keypoints:
        csv_headers.extend([i+'_x',i+'_y'])
    csv_headers.extend(["x_max",'x_min','y_max','y_min'])

    flag=False

    # 逐行读取数据
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        pic_name=row['image_name']
        pic_ele=pic_name.split('/')[1].split('-')
        pic_id,pic_cond,pic_cond_id,pic_view=pic_ele
        # print(pic_id,pic_cond,pic_cond_id,pic_view)
        if pre_pic_id!="" and (pre_pic_id!=pic_id or pre_pic_cond!=pic_cond or pre_pic_cond_id!=pic_cond_id or pre_pic_view!=pic_view):
            csv_output_filename='./testout2/'+pre_pic_id+'/'+pre_pic_cond+'-'+pre_pic_cond_id+'/'+'pose'+str(pre_pic_view)+'.csv'
            print(pre_pic_id+'-'+pre_pic_cond+'-'+pre_pic_cond_id)
            if pre_pic_cond_id=='01' and pre_pic_id=='075':
                plot_a_pic(csv_output_filename,keypoints_list, True if int(pre_pic_view)<90 else False)

            keypoints_list.clear()
            resize_scale_list.clear()
        

        # # 创建一个字典来存储每个关键点的坐标
        # keypoints_dict = {'image_name': row['image_name']}
        pre_pic_id=pic_id
        pre_pic_cond=pic_cond
        pre_pic_cond_id=pic_cond_id
        pre_pic_view=pic_view
        temp_x_min=fig_wid
        temp_x_max=0
        temp_y_min=fig_height
        temp_y_max=0
        keypoints_list.append(list())
        temp_keypoints_list=list()
        for keypoint in keypoints:
            x = row[f'{keypoint}_x']
            y = row[f'{keypoint}_y']
            temp_x_min=min(x,temp_x_min)
            temp_x_max=max(x,temp_x_max)
            temp_y_max=max(y,temp_y_max)
            temp_y_min=min(y,temp_y_min)
            temp_keypoints_list.append([x,y])

        frame_height=temp_y_max-temp_y_min
        frame_wid=temp_x_max-temp_x_min
        h_=target_height*(N-2)/N
        w_=frame_wid*h_/frame_height
        # w_=frame_wid*target_height/frame_height
        for i in temp_keypoints_list:
            x_=h_*(i[0]-temp_x_min)/frame_height+(512/2-w_/2)   
            y_=h_*(i[1]-temp_y_min)/frame_height+target_height/N
            # y_=h_+h_*(temp_y_min-i[1])/frame_height+tar
            keypoints_list[-1].append([x_,y_])
        resize_scale_list.append([temp_x_max,temp_x_min,temp_y_max,temp_y_min])
        # if index>=200:
        #     break