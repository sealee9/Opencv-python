# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:42:07 2019

@author: Administrator
"""

'''
图像梯度、图像边界等
函数有：cv2.Sobel(),cv2.Scharr(),cv2.Laplacian()等
'''
'''
原理： 梯度简单来说就是求导
opencv提供了上述的三种梯度滤波器，也就是高通滤波器，
Sobel，Scharr是求一阶或二阶导数，Scharr是对Sobel的优化，Laplacian是求二阶导
使用3X3滤波器时，尽量使用Scharr,速度与Sobel相同，可是效果更好
当cv2.Sobel()中ksize这个参数设为-1时，会默认使用Scharr函数
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('balck_white.jpg')
#对于输出图像的深度设置为cv2.CV_64F
#是因为：如果设为-1与原图像保持一致，若原图像为np.uint形式，
#在计算梯度时，从白到黑的梯度是负数，此时的边界将显示不出来。
laplacian = cv2.Laplacian(img,-1)
#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobel_x = cv2.Sobel(img,-1,1,0,ksize=5)
sobel_y = cv2.Sobel(img,-1,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobel_x,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel_y,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.show()









