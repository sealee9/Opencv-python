# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:35:30 2019

@author: Administrator
"""

'''
边缘检测概念
函数cv2.Canny()

步骤：
1、由于边缘检测很容易受到噪声的影响，所以首先使用5x5的高斯滤波器去除噪声
2、对平滑后地图像使用sobel算子计算x,y方向的梯度Gx，Gy，根据Gx，Gy找到边界的梯度和方向
        G=Gx，Gy两者的平方和再开方
        方向等于Gx/Gy的反正切函数值
        方向被归为四类：垂直、水平、两个对角线
3、非极大值抑制
        在获得梯度大小和方向后，应对图像进行扫描，去除非边界上的点，对每一个像素进行检查，查看该点的
        梯度是不是周围具有相同梯度方向的点中最大的
4、滞后阈值：确定真正的边界
        设置两个阈值，minVal、maxVal,当图像的灰度值大于maxVal被认为是真正的边界，小于minVal将被抛弃
        若介于两者之间时，判断是否与某个确定Wie真正边界点相连，若相连则是边界点。
opencv提供了cv2.Canny()函数完成上面所有的步骤
参数：1 输入图像，2 minVal,3 maxVal,4 sobel算子的卷积核大小，默认值时3，最后一个参数：L2gradient
最后一个参数用来设定梯度大小的计算公式，若为True,则计算公式为上述所列
若为False，则为Gx，Gy两者的平方和，默认值是False
'''
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('black_white.jpg',0)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)
plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap='gray')
plt.xticks([]),plt.yticks([])













