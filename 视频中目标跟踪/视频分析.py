# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:21:41 2019

@author: Administrator
"""

'''
Meanshift和Camshift

目标: 学习使用Meanshift和Camshift算法在视频中找到并跟踪目标对象

Meanshift算法的基本原理：  假设我们有一堆点（比如直方图反向投影得到的点）和一个小的圆形窗口，我们
                         要完成的任务就是将这个窗口移动到最大灰度密度处(或者点最多的地方)
                         初始窗口是一个圆C1，圆心为C1_o,但是这个圆形窗口中所有点的质心却是C1_r，
                         这是因为这个圆形包含区域的点分布不均匀导致圆心和所有点的质心不重合，所以我们
                         要移动圆心C1_o到质心C1_r，这样我们会得到一个新的窗口区域，大多数情况下质心和圆心
                         还是不重合，所以重复上面的操作，将新窗口的中心移动到新的质心，就这样不挺的迭代操作
                         直到窗口的中心和其所有包含点的质心重合为止(或者有一点小误差),按照这样的操作我们的
                         窗口最终会落在像素值(和)最大的地方。
通常情况下我们要使用直方图反向投影得到的图像和目标对象的起始位置，当目标对象的移动会反映到直方图反向投影图中
这样，meanshift算法就把我们的窗口移动到图像中灰度密度最大的区域。

Opencv中的Meanshift：
                    要在opencv中使用meanshift算法，首先我们要对目标对象进行设置，计算目标对象的直方图
                    这样在执行meanshift算法时我们就可以将目标对象反向投影到每一帧中去了，另外还需要提供
                    窗口的起始位置。在这里计算H(Hue)通道的直方图，同样为了避免低亮度造成的影响，我们要
                    使用函数cv2.inRange()将低亮度的值忽略掉

'''

'''
import numpy as np
import cv2

cap = cv2.VideoCapture('../Opencv/images/slow.mp4')

#获取视频第一帧
ret,frame = cap.read()
#设置窗口初始位置
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((188.,255,255)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while(1):
    ret,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        ret,track_window = cv2.meanShift(dst,track_window,term_crit)
        
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),255,2)
        cv2.imshow('img2',img2)
        
        k = cv2.waitKey(60)
        if k == 27:
            break
        #else:
         #   cv2.imwrite(chr(k)+".jpg",img2)
            
    else:
        break
cv2.destroyAllWindows()
cap.release()
''' 


#############################################################################
#############################################################################

#Camshift算法
'''
在上面的meanshift中，窗口的大小是固定的，汽车由远到近在视觉上是一个逐渐变大的过程，固定
的窗口是不合适的，需要根据目标的大小和角度来对窗口的大小和角度进行修订，opencvLabs提供了
解决方案，CAMshift算法

这个算法首先使用meanshift算法找到目标后，再去调整窗口的大小，它还会计算目标对象的最佳外接圆的
角度，并以此调节窗口角度，然后使用新的窗口大小和角度在原来的位置继续进行meanshift，重复这个过程
直到达到需要的精度。
与meanshift基本一样，但是返回的结果是一个带旋转角度的矩形，以及这个矩形参数被用到下一次迭代中

'''
import numpy as np
import cv2

cap = cv2.VideoCapture('../Opencv/images/slow.mp4')

#获取视频第一帧
ret,frame = cap.read()
#设置窗口初始位置
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((188.,255,255)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while(1):
    ret,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        ret,track_window = cv2.CamShift(dst,track_window,term_crit)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True,255,2)
        cv2.imshow('img2',img2)
        
        k = cv2.waitKey(60)
        if k == 27:
            break
        #else:
         #   cv2.imwrite(chr(k)+".jpg",img2)
            
    else:
        break
cv2.destroyAllWindows()
cap.release()














   

        
        
        
        
        
        
    
    
    
    





















