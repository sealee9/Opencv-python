# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:27:47 2019

@author: Administrator
"""

'''
作用：可以用来做图像分割或者在图像中找到感兴趣的部分
它会输出与输入图像（待搜索）同样大小的图像，其中图像上每一个像素值代表了输入图像上
对应点属于目标对象的概率

另一种解释：输出图像中像素值越高（越白）的点越可能代表我们要搜索的目标
直方图投影经常与camshift算法一起使用

实现：    1.首先为一张包含查找目标的图像建立直方图，查找对象尽量占满整张图像，
            比如我们要在messi这张图像上找到草地区域，先找另一张图片，这个图片只包含草地
            然后计算这张草地的彩色直方图     
           （ 因为一个物体的颜色要比灰度更好的被用来进行分割和识别）
         2.把颜色直方图先归一化
           然后使用cv2.calcBackProject函数计算，也就是找到输入图像中每一个像素点的像素值在直方图中对应的概率
         3.上面操作得到一个概率图像，最后设置适当的阈值对概率图像进行二值化
         4.把二值化的图像和与图像进行位与，得到我们需要的结果

'''

import cv2
import numpy as np

img_org = cv2.imread('messi.jpg')
img_tar = cv2.imread('ground.jpg')

#把图像转换到 HSV格式
hsv_org = cv2.cvtColor(img_org,cv2.COLOR_BGR2HSV)
hsv_tar = cv2.cvtColor(img_tar,cv2.COLOR_BGR2HSV)

#计算要搜索的目标的直方图，之后要将原图像映射到这个直方图求概率
hist_tar = cv2.calcHist([hsv_tar],[0,1],None,[180,256],[0,180,0,256])

#归一化
cv2.normalize(hist_tar,hist_tar,0,255,cv2.NORM_MINMAX)

#直方图反向投影，第一个参数是原图像，要有[]，第三个参数是目标的直方图
dst = cv2.calcBackProject([hsv_org],[0,1],hist_tar,[0,180,0,256],1)
#上面求得的结果是原图像的点是目标点的概率，值是0到255,值越大的点是目标点的概率越大


#利用圆形卷积核把分散的点连在一起
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,disc)

#二值化
ret,thresh = cv2.threshold(dst,50,255,0)

#因为原图像是彩色的，是三通道，所以要merge
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(img_org,thresh)

cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()





'''在numpy中实现直方图反向投影'''
import cv2
import numpy as np

img_org = cv2.imread('messi.jpg')
img_tar = cv2.imread('ground.jpg')

#把图像转换到 HSV格式
hsv_org = cv2.cvtColor(img_org,cv2.COLOR_BGR2HSV)
hsv_tar = cv2.cvtColor(img_tar,cv2.COLOR_BGR2HSV)

#计算直方图
hist_org = cv2.calcHist([hsv_org],[0,1],None,[180,256],[0,180,0,256])
hist_tar = cv2.calcHist([hsv_tar],[0,1],None,[180,256],[0,180,0,256])
#计算目标的彩色直方图与的原图像的彩色直方图比值
R = hist_tar/hist_org

#分离出原图像HSV中的HSV
h,s,v = cv2.split(hsv_org)

#计算原图上对应的点是目标上点的概率
B = R[h.ravel(),s.ravel()]
print(B.shape)

B = np.minimum(B,1)
B = B.reshape(hsv_org.shape[:2])
print(B)

#利用圆形卷积核把分散的点连在一起
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
B = cv2.filter2D(B,-1,disc)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)

#二值化得到感兴趣区域
ret,thresh = cv2.threshold(B,50,255,0)

#因为thresh是单通道，而原图像是三通道，所以要merge才能和原图像融合
thresh = cv2.merge((thresh,thresh,thresh))

#与原图像融合
res = cv2.bitwise_and(img_org,thresh)

cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()





























































