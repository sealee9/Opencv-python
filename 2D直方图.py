# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:09:10 2019

@author: Administrator
"""

'''
之前绘制的直方图只考虑了图像的灰度值一个特性，只有一维
当要考虑图像的两个属性：颜色(Hue)和饱和度(Saturation)时，
绘制2D直方图
使用opencv统计直方图
函数：cv2.calcHist()
首先要将图像从BGR转换成HSV
然后使用函数统计直方图
channels = [0,1],因为有两个通道H，S
bins = [180,256],H通道为180，S为256
range = [0,180,0,256],H的取值范围为0-180，S为0-256
'''
#代码：
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('apple.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
plt.imshow(hist,interpolation='nearest')
plt.show()


'''numpy中计算2D直方图'''
h = hsv[:,:,0]
s = hsv[:,:,1]
hist,xbins,ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])
plt.imshow(hist,interpolation='nearest')
plt.show()

# 还可以使用cv2来显示直方图,绘制的图是灰度图，与上面绘制的结果不一样
cv2.imshow('histogram',hist)
cv2.waitKey(0)
cv2.destroyAllWindows()

















