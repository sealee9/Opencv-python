# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:20:51 2019

@author: Administrator
"""

'''
如何对图像进行颜色空间转换，从BGR到灰度，或者从BGR到HSV等
创建一个程序从一副图像中获取特定颜色的物体
函数有:cv2.cvtColor(),cv2.inRange()等

'''
'''
转换颜色空间，opencv中有超过150中颜色空间转换方法，经常用到的是BGR->GRAY，和BGR->HSV
用到的函数是cv2.cvtColor(input_image,flag),flag是转换类型
BGR->GRAY，flag是cv2.COLOR_BGR2GRAY
BGR->HSV,flag是cv2.COLOR_BGR2HSV
注意：在opencv中HSV格式中，H（色彩/色度）取值范围是[0,179],S(饱和度)范围是[0,255]
V(亮度)范围是[0,255]，在不同软件中，三者范围可能不一样，所以在对比不同软件中的HSV时，首先
要进行归一化在进行比较
'''

#查看转化方法
import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)    #有274种

'''
获取一张图像中的蓝色部分物体
在HSV空间比在BGR中更容易获取有特定颜色
'''
import cv2
import numpy as np

bgr = cv2.imread('test.jpg')
#将图片转换
hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
#设定蓝色阈值
low_blue = np.array([90,100,100])
high_blue = np.array([130,255,255])
#根据阈值建掩模
mask = cv2.inRange(hsv,low_blue,high_blue)
#将掩模和原图像进行位运算
res=cv2.bitwise_and(bgr,bgr,mask=mask)
cv2.imshow('bgr',bgr)
cv2.waitKey(0)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''找到BGR中各个颜色对应的HSV中的值'''
import cv2
import numpy as np
'''注意是三括号'''
blue = np.uint8([[[255,0,0]]]) 
hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
print(hsv_blue)



'''同时在一张图像中提取红绿蓝三种颜色物体'''
import cv2
import numpy as np
#将图片转换
bgr = cv2.imread('rgb.png')
hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
#设定三色阈值
#参考上面上面的程序
low_blue = np.array([120,100,100])
high_blue = np.array([130,255,255])
low_green = np.array([50,100,100])
high_green = np.array([70,255,255])
low_red = np.array([0,100,100])
high_red = np.array([20,255,255])
#根据阈值建掩模
mask1 = cv2.inRange(hsv,low_blue,high_blue)
mask2 = cv2.inRange(hsv,low_green,high_green)
mask3 = cv2.inRange(hsv,low_red,high_red)
mask = cv2.add(mask1,mask2)
mask = cv2.add(mask,mask3)
cv2.imwrite('balck_white.jpg',mask)
#将掩模和原图像进行位运算
res=cv2.bitwise_and(bgr,bgr,mask=mask)
cv2.imshow('bgr',bgr)
cv2.waitKey(0)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''几何变换'''
'''扩展缩放，缩放时使用插值方法cv2.INTER_AREA，扩展使用cv2.INTER_CUBIC或者cv2.INTER_LINEAR'''
import cv2
import numpy as np

img = cv2.imread('cat.jpg')
#由于后面设置了缩放因子，所以输出图像的参数可以使None
res = cv2.resize(img,None,fx=1.5,fy=1,interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)


import cv2
import numpy as np

img = cv2.imread('cat.jpg')
height,width=img.shape[:2]
res = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)



'''平移'''
'''平移矩阵为：[[1,0,Tx],[0,1,Ty]],Tx,Ty为平移距离'''
'''cv2.warpAffine()接受的平移矩阵是2x3'''
import cv2
import numpy as np

img = cv2.imread('cat.jpg')
height,width=img.shape[:2]
M = np.array([[1,0,200],[0,1,100]],dtype=np.float32)
res = cv2.warpAffine(img,M,(width,height)) #输入图像，平移矩阵，输出图像大小
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''旋转图像'''
import cv2
import numpy as np

img = cv2.imread('cat.jpg')
height,width=img.shape[:2]
#通过cv2.getRotationMatrix2D()函数得到旋转矩阵，参数为旋转中心，旋转角度，以及缩放大小
M = cv2.getRotationMatrix2D((width/2,height/2),45,0.6)
res = cv2.warpAffine(img,M,(width,height)) #输入图像，平移矩阵，输出图像大小
cv2.imshow('image',res)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)

'''仿射变换，原来图像中平行的线在变化后的图像依然平行'''
'''需要在前后图像中找三个点'''
import cv2
import numpy as np
img = cv2.imread('cat.jpg')
height,width=img.shape[:2]
pts1 = np.float32([[50,50],[200,50],[50,200]])  #原图像中选取的点
pts2 = np.float32([[10,100],[200,50],[100,250]])  #选取的点在变换图像中存在的位置
M = cv2.getAffineTransform(pts1,pts2)   #2*3矩阵
res = cv2.warpAffine(img,M,(width*2,height*2)) #输入图像，平移矩阵，输出图像大小
cv2.imshow('image',res)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)
        
'''透视变换，需要一个3x3矩阵，直线变换后还是直线，需要找原图中的四个点和变换后在变换
图像中四个点，任意三个点不共线'''

import cv2
import numpy as np
img = cv2.imread('cat.jpg')
print(img)
height,width=img.shape[:2]
pts1 = np.float32([[0,0],[927,0],[0,682],[927,628]])  #原图像中选取的点
pts2 = np.float32([[0,200],[200,0],[0,682],[200,380]])  #选取的点在变换图像中存在的位置
M = cv2.getPerspectiveTransform(pts1,pts2)   #3*3矩阵
print(M)
res = cv2.warpPerspective(img,M,(927,682)) #输入图像，变换矩阵，输出图像大小
cv2.imshow('image',res)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)










