# -*- coding: utf-8 -*-
import os,dlib,glob #sys,
import numpy as np
from skimage import io
#import imutils
import cv2

"""if len(sys.argv) != 2:
    print("請重新確認要辨識的圖片名稱")
    exit()"""

# 人臉圖片資料夾名稱
faces_data_path="./resources"
# 要辨識的圖片名稱
"""img_name=sys.argv[1]"""
# 載入人臉檢測器
detector=dlib.get_frontal_face_detector()
# 人臉68特徵點模型的路徑及檢測器
shape_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 載入人臉辨識模型及檢測器
face_rec_model=dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 人臉描述子list
descriptors = []
# 候選人臉名稱list
candidate=[]

for file in glob.glob(os.path.join(faces_data_path, "*.jpg")):
    base=os.path.basename(file)
    # 讀取人臉圖片資料夾的每張圖片
    # os.path.join()用於拼接檔案路徑
    # os.path.splittext()分離檔名及副檔名
    
    candidate.append(os.path.splitext(base)[0])
    img=io.imread(file)
    
    # 人臉偵測
    dets=detector(img,1)
    
    for k,d in enumerate(dets):
        # 68特徵點偵測
        shape=shape_predictor(img,d)
        
        # 128維特徵向量描述子
        face_descriptor=face_rec_model.compute_face_descriptor(img,shape)
        
        # 轉換numpy array格式
        v=np.array(face_descriptor)
        descriptors.append(v)

imgs=["01","02","03","20181012132354"]
qq=0

for x in imgs:
    
    img_name=x+".jpg"
    # 對要辨識的目標圖片做相同處理
    # 讀取圖片
    img=io.imread(img_name)
    # 人臉偵測
    dets=detector(img,1)
    
    #distance=[]
    
    for k,d in enumerate(dets):
        distance=[]
        # 68特徵點偵測
        shape=shape_predictor(img,d)    
        # 128維特徵向量描述子
        face_descriptor=face_rec_model.compute_face_descriptor(img,shape)
        # 轉換numpy array格式
        d_test=np.array(face_descriptor)
        
        x1=d.left()
        y1=d.top()
        x2=d.right()
        y2=d.bottom()
        # 以方框框出人臉
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1,cv2.LINE_AA)
        
        # 計算歐式距離，線性運算函式
        for i in descriptors:
            dist_=np.linalg.norm(i-d_test)
            distance.append(dist_)
    
        # 利用zip函數將元素打包成一個元組
        # 並存入dict (候選圖片,距離)
        candidate_distance_dict=dict(zip(candidate,distance))
        
        # 接著將候選圖片及人名進行排序
        candidate_distance_dict_sorted=sorted(candidate_distance_dict.items(),key=lambda d:d[1])
        # 最短距離為辨識出的對象
        result=candidate_distance_dict_sorted[0][0]
        print("result = {}".format(result))
        # 在方框旁邊標上人名
        cv2.putText(img,result,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1,cv2.LINE_AA)
    
    #img=imutils.resize(img,width=500)
    qq=qq+1
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("outcome{}".format(qq),img)

cv2.waitKey(0)
cv2.destroyAllWindows()







