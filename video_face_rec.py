# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:28:34 2020

@author: 169151
"""

import os,dlib,glob #sys,
import numpy as np
from skimage import io
import cv2
#import imutils

# 開啟影片檔案
cap = cv2.VideoCapture('test.mp4')

# 取得畫面尺寸
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/1.5)

# 使用 XVID 編碼
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (width, height))

# 人臉圖片來源資料夾名稱
faces_data_path="./resources"

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()
shape_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 載入人臉辨識模型及檢測器
face_rec_model=dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 人臉描述子list
descriptors = []
# 候選人臉名稱list
candidate=[]

# 從圖片庫建立人臉特徵向量
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

cv2.namedWindow("Face Detection",cv2.WINDOW_NORMAL)
# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
    
    ret, frame = cap.read()
    # 如果 ret 返回 True 表示有畫面，False 表示沒有畫面則結束迴圈
    if ret==True:
        frame = cv2.resize(frame, (width, height))
            
        # 偵測人臉
        face_rects, scores, idx = detector.run(frame, 0)
      
        # 取出所有偵測的結果
        for i, d in enumerate(face_rects):
            distance=[]
            # 68特徵點偵測
            shape=shape_predictor(frame,d)    
            # 128維特徵向量描述子
            face_descriptor=face_rec_model.compute_face_descriptor(frame,shape)
            # 轉換numpy array格式
            d_test=np.array(face_descriptor)
            
            
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            #text = "%2.2f(%d)" % (scores[i], idx[i])
        
            # 以方框標示偵測的人臉
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
            
            # 計算歐式距離，線性運算函式
            for j in descriptors:
                dist_=np.linalg.norm(j-d_test)
                distance.append(dist_)
        
            # 利用zip函數將元素打包成一個元組
            # 並存入dict (候選圖片,距離)
            candidate_distance_dict=dict(zip(candidate,distance))
            
            # 接著將候選圖片及人名進行排序
            candidate_distance_dict_sorted=sorted(candidate_distance_dict.items(),key=lambda d:d[1])
            # 最短距離為辨識出的對象
            result=candidate_distance_dict_sorted[0][0]
            print("result = {}".format(result))
        
            # 標示分數
            cv2.putText(frame, result, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 0, 255), 1, cv2.LINE_AA)
                

        # 寫入影格
        out.write(frame)           

        # 顯示結果
        cv2.imshow("Face Detection", frame)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
        print("辨識結束")
        break

cap.release()
# out.release()
cv2.destroyAllWindows()