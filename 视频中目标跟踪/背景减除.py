# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:57:36 2019

@author: Administrator
"""

'''
基础：
        在很多基础应用中，背景减除是一个非常重要的步骤，例如顾客统计，使用一个静态摄像头来记录进去和离开
        房间的人数，或者是交通摄像头，需要提供交通工具的信息等，在所以的这些例子中，首先要将人或者车辆单独
        提取出来，技术上来说，需要从静止的背景中提取移动的前景。
        如果你有一张背景(仅有背景不含前景)图像，比如没有顾客的房间，没有交通工具的道路等，那就好办，只需要
        在新的图像中减去背景就可以得到前景对象了。但在大多数情况下，没有这样的背景图像，所以需要从有的图像
        中提取前景，如果图像中的交通工具还有影子的话，那工作就更难了，因为影子在移动，仅仅使用减法会把影子
        也当成前景，这是一件很复杂的事。

BackgroundSubtractorMOG:
                            这是一个以混合高斯模型为基础的前景/背景分割法，是在2001年提出来的，它使用
                            K(k=3或5)个高斯分布混合对背景像素进行建模，使用这些颜色(在整个视频中)存在
                            时间的长短作为混合的权重，背景的颜色一般持续的时间最大，而且更加静止。
                            背景建模是基于时间序列的，每一个像素点所在的位置在整个时间序列中会有很多值，
                            从而构成一个分布。
                            函数cv2.creatBackgroundSubtractorMOG()创建一个背景对象，这个函数有些
                            可选参数，比如：进行建模场景的时间长度，高斯混合成分的数量，阈值等，将它们
                            全部设置为默认值，然后在整个视频中需要使用backgroundSubtractor.apply()
                            就可以得到前景的掩模了。
                            该版本的opencv中已经没有cv2.createBackgroundSubtractorMOG()
                            但是有cv2.createBackgroundSubtractorMOG2()
                                                                                    

BackgroundSubtractorMOG2:
                            这个算法大部分与前一个相似，这个算法的特点是它为每一个像素选择一个合适数目
                            的高斯分布(前一个算法使用的是K高斯分布)，这样就会对由于亮度等发生变化引起
                            的场景变化产生更好的适应性，
                            和前面一样需要创建一个背景对象，但在这里可以选择是否检测阴影，如果
                            detectShadows=True（默认值），就会检测影子并将影子标记出来，但是
                            这样做会降低处理速度，影子会被标记成灰色。
'''

import numpy as np
import cv2

cap = cv2.VideoCapture('../Opencv/images/vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret,frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

'''
BackgroundSubtractorGMG：
                            此算法结合了静态背景图像估计和每个像素的贝叶斯分割，这是2012年提出来的，
                            它使用前面很少的图像(默认是前120帧)进行背景建模，使用了概率前景估计算法
                            (使用贝叶斯估计鉴定前景),这是一种自适应的估计，新观察到的对象比旧的对象
                            具有更高地权重，从而对光照变化产生适应，一些形态学操作如开运算闭运算等被
                            用来除去不需要的噪声，在前几帧图像中你会得到一个黑色窗口，对结果进行形态
                            学开运算与闭运算对除去噪声很有帮助。
                            该版本opencv中没有BackgroundSubtractorGMG
'''

import numpy as np
import cv2

cap = cv2.VideoCapture('../Opencv/images/vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG()
while(1):
    ret,frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    








'''
KNN背景分割器，cv2.createBackgroundSubtractorKNN
'''

import cv2
import numpy as np

camera = cv2.VideoCapture('../Opencv/images/slow.mp4') # 参数0表示第一个摄像头
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while True:
    grabbed, frame_lwpCV = camera.read()
    fgmask = bs.apply(frame_lwpCV) # 背景分割器，该函数计算了前景掩码
    # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # 下面就跟基本运动检测中方法相同，识别目标，检测轮廓，在原始帧上绘制检测结果
    dilated = cv2.dilate(th, es, iterations=2) # 形态学膨胀
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('mog', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

















