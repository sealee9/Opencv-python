# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:06:12 2019

@author: Administrator
"""

'''
目标：  
        1、光流的概念以及Lucas-Kanade光流法
        2、使用函数cv2.calcOpticalFlowPyrLK()对图像中的特征点进行跟踪
光流：
        由于目标对象或者摄像机的移动造成的图像对象在连续两帧图像中的移动被称为光流，它是
        一个2D向量场，可以用来显示一个点从第一帧图像到第二帧图像之间的移动。
光流的应用领域:由运动重建结构、视频压缩、Vedio Stabilization等

光流是基于以下假设的：
                    1、在连续的两帧图像之间(目标对象的)像素的灰度值不改变
                    2、相邻的像素具有相同的运动
Lucas-Kanade算法：
                    根据第一条假设，领域内所有点都有相似的运动，Lucas-Kanade算法就是了利用
                    一个3x3领域中的9个点具有相同运动的这一特点，就可以找到9个点的光流方程
                    (根据第一个假设只能找到一个等式但有两个未知数，解不开方程),九个点组成
                    一个具有两个未知数的9个等式的方程组，这是一个约束条件过多的方程组，一个
                    好的解决方法就是使用最小二乘拟合。
                    我们目前处理的都是很小的运动，对于大的运动，可以使用图像金字塔的顶层，
                    此时小的运动被移除，大的运动换成了小的运动，再试用Lucas-kanade法，就会
                    得到尺度空间上的光流。

Opencv中的Lucas-Kanade光流：
                             上述所有过程被opencv打包成一个函数：cv2.calcOpticalFlowPyrLK()
                             首先使用函数cv2.goodFeatureToTrack()来确定要跟踪的点，先在视频
                             的第一帧图像中检测到一些Shi-Tomasi角点，然后使用Lucas-Kanade算法
                             迭代跟踪这些角点，要给函数cv2.calcOpticalFlowPyrLK()传入前一帧
                             图像和其中的点，以及下一帧图像，函数将返回带有状态数的点，如果状态数
                             是1，说明在下一帧图像中找到了这个点(上一帧中检测到的角点),如果状态数
                             是0，说明没有在下一帧中找到这个点，再把这些点作为参数传到函数，如此迭代
                             下去实现跟踪
                             (上面的代码没有对返回的角点的正确性进行检查，图像中的一些特征点甚至在丢失
                             以后，光流还会找到一个预期相似的点，所以为了实现稳定的跟踪，应该每个一定
                             间隔就要进行一次角点检测。)

'''
'''
import numpy as np
import cv2

cap = cv2.VideoCapture('../Opencv/images/slow.mp4')
#shitomasi角点检测参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
#lucas-kanade光流算法参数
#maxLevel为使用的图像金字塔层数
lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
#创建一些随机颜色
color = np.random.randint(0,255,(100,3))

ret,old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    pl,st,err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
    #选择好的点
    good_new = pl[st==1]
    good_old = p0[st==1]
    #绘制跟踪
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    
    cv2.imshow('frame',img)
    k = cv2.waitKey(30)
    if k == 27:
        break
    
    #更新前一帧和前一帧的点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
    
'''


'''
Opencv中的稠密光流：
                    Lucas-Kanade法是计算一些特征点的光流，上面的例子是使用Shi-Tomasi算法检测到的角点
                    opencv中还提供了一种计算稠密光流的方法，它会绘制图像中所有点的光流，这是基于
                    Gunner_Farneback算法的（2003年）
                    算法的结果是一个带有光流的向量(u,v)的双通道数组，通过计算我们能得到光流的大小和方向
                    使用颜色对结果进行编码以便于更好的观察，方向对应于H(Hue)通道，大小对应于V(Value)
                    通道。                    
'''

import cv2 
import numpy as np

cap = cv2.VideoCapture('../Opencv/images/vtest.avi')

ret,frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret,frame2 = cap.read()
    next_f = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #函数cv2.calcOpticalFlowFarneback的参数说明
    #1、前一帧图片
    #2、下一帧图片
    #4、以多大尺度建立图像金字塔，如果为0.5，表示金字塔下一层图片是前一层的二分之一
    #5、金字塔层数
    #6、winsize
    #7、迭代次数
    #8、每一个像素点使用的邻域大小
    #9、邻域系数，邻域是5时，值为1.2，邻域为7时，值为1.5
    #10、flag,为0时，计算快，为1，时，慢但准确
    flow = cv2.calcOpticalFlowFarneback(prvs,next_f,None,0.5,3,15,3,5,1.2,0)
    
    mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30)
    
    if k==27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next_f
cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    


















  
    
    
    
        
        
        
        
        
        
        
        
        
        
    
    
    
    


































