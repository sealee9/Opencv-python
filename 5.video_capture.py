# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:02:22 2019

@author: Administrator
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.open(0)
while(True):
    #一帧一帧的捕捉视频信息
    ret,frame = cap.read()
    #将捕捉到的图片转换成灰度
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow(frame,gray)
    if cv2.waitKey(1):
        break
cap.release()
cv2.destroyAllWindows()
cv2.VideoCapture.open()