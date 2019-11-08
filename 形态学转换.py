# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:10:07 2019

@author: Administrator
"""

'''
形态学操作：腐蚀，膨胀，开运算，闭运算
函数：cv2.erode(),cv2.dilate(),cv2.morphologyEx()
'''

'''
腐蚀操作
建立一个核，核在图像上逐个像素滑动，若卷积核对应原图区域上像素值都是1，那么这个中心点为1，否则为0
所以腐蚀会使白色的情景物体变小，可以用来消除白噪声，分开两个连在一起的白色前景物体
'''
import cv2
import numpy as np

img_1 = cv2.imread('circles.png')
print(img_1.shape)
kernel = np.ones((3,3),np.uint8)
img = cv2.erode(img_1,kernel,iterations=3)
cv2.imshow('original',img_1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
膨胀操作跟腐蚀操作相反
'''
img = cv2.dilate(img_1,kernel,iterations=1)
cv2.imshow('original',img_1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
开运算
先腐蚀再膨胀
用来去除噪声
'''

res1 = cv2.morphologyEx(img_1,cv2.MORPH_OPEN,kernel)
cv2.imshow('开运算',res1)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
闭运算
用来填充前景物体的空洞和黑点
'''
cv2.morphologyEx(img_1,cv2.MORPH_CLOSE,kernel)

'''
形态学梯度
与腐蚀和膨胀差别开来
结果看起来是前景图像的轮廓
空心图像
:膨胀-腐蚀
'''
res2 = cv2.morphologyEx(img_1,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('gradient',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''礼帽：原图像-开运算'''

res3 = cv2.morphologyEx(img_1,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('tophat',res3)
cv2.imshow('bijiao',res1-img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''黑帽：闭运算-原图像'''
res = cv2.morphologyEx(img_1,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('tophat',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
结构化元素
上面的操作使用的结构因子都是正方形的
cv2.getStructionElement()构建不同形状的结构因子
只需要传递结构因子的形状和大小
'''
#矩形
shape_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
print(shape_1)
#椭圆
shape_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print(shape_2)
#十字形
shape_3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
print(shape_3)








