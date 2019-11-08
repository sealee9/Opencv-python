# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:19:07 2019

@author: Administrator
"""

'''
获取像素值并修改
获取图像的属性
图像的ROI图像通道的拆分与合并
'''

import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
print(img.shape)
px = img[100,100] #获取的是图像上某点的三通道的值
print(px)
px_sub = img[100,100,2] #获取的是图像上某点的其中一个通道的值
print(px_sub)
 #修改点的像素值
img[100,100]=[10,10,10]
print(img[100,100])

'''上面是使用矩阵的切片来获取像素值的，
要想获取bgr单通道值，可使用numpy中的array.item()和array.itemset()'''
import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
print(img.item(100,100,1))
img.itemset((100,100,1),100)
print(img.item(100,100,1))

'''图像属性'''
import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
print(img.shape)  #获取行、列和通道数
print(img.size) #获取整个图像上的像素数
print(img.dtype) #返回数据类型


import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
r = img[600:650,200:250]
img[100:150,800:850] = r
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
img[:,:,1]=0
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''拆分和合并通道'''

import numpy as np 
import cv2
img = cv2.imread('cat.jpg')
b,g,r = cv2.split(img)
print(b.shape)
img = cv2.merge((b,g,r))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''使用split比较耗时间，如果是只是想将对应通道重新赋值，选择numpy的索引，img[:,:,0]=赋值
'''

'''图像加法操作,cv2.add()将两幅图像进行加法运算是一种饱和运算，例如和大于255按255替代
也可以用numpy加法运算，是一种模运算
'''
import cv2
import numpy
a = np.uint8([250])
b = np.uint8([10])
print(cv2.add(a,b))  #250+10=260>255,      所以为[[255]]
print(a+b)       #250+10=260>256,260%256=4        [4]

'''将两幅图像混合，要保证图像的大小格式等一样'''
import numpy as np 
import cv2
img1 = cv2.imread('bike.jpg')

img2 = cv2.imread('bottle.jpg')
img = cv2.addWeighted(img1,0.7,img2,0.3,0)
#img = cv2.add(img1,img2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''耗费时间计算'''
import numpy as np
import cv2

img = cv2.imread('cat.jpg')
e1 = cv2.getTickCount()
for i in range(5,10,2):
    img = cv2.medianBlur(img,i)
e2 = cv2.getTickCount()
t = (e2-e1)/cv2.getTickFrequency()
print(t)

import time
import numpy as np
import cv2

img = cv2.imread('cat.jpg')
e1 = time.time()
for i in range(5,20,2):
    img = cv2.medianBlur(img,i)
    cv2.imshow('image',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
e2 = time.time()
t = e2-e1
print(t)


