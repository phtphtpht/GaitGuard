import cv2
# cap=cv2.VideoCapture('/home/xk/pht/gaitfake/yolo/data/casiab_gen/077/077-nm-01-000.mp4')
cap=cv2.VideoCapture('/home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/assets/wxdwalk_kps.mp4')
for i in range(200):
    ret,frame=cap.read()
    if i %5==0:
        cv2.imwrite('./draw_pic/'+str(i)+'_wxd.jpg',frame)
cap.release()