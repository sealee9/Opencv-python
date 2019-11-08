# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:56:23 2019

@author: Administrator
"""

'''
轮廓概念
寻找轮廓、绘制轮廓
函数：cv2.findContours(),cv2.drawContours()

轮廓：可以简单认为是沿着边界连续的点形成的曲线，具有相同的颜色或者灰度
为了更加准确，在寻找轮廓之前，要二值化图像或者Canny边缘检测
查找轮廓的函数会修改原始图像，如果在查找轮廓之后还使用原图像，需要在查找之前将原图像数据保存在另一个变量中
在opencv中查找轮廓就像在黑色背景中寻找白色物体

在一个二值化图像中查找轮廓
cv2.findContours(),第一个参数：输入图像；第二个参数：轮廓检索模式；第三个：轮廓近似方法
返回值有三个：1.图像，2.轮廓，是一个python列表，存储这个图像中的所有轮廓，每个轮廓是一个numpy数组，包含对象边界点的坐标（x,y）
第三个：轮廓的层析结构
'''
'''
绘制轮廓，使用函数cv2.drawContours()
第一个参数：原始图像，第二个参数：轮廓（一个python列表），第三个：轮廓索引，-1为所有轮廓，接下来是轮廓颜色和厚度
'''




import cv2
import numpy as np
img=cv2.imread('cat.jpg')
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
image = cv2.drawContours(img,contours,-1,(0,255,0),2)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
关于cv2.findContours()函数中的第三个参数轮廓近似方法，在寻找轮廓时，我们寻找的具有相同灰度值的边界点
轮廓会储存边界点的坐标x,y，但是边界点非常多，需要全部储存吗？
cv2.CHAIN_APPROX_NONE存储所有的点
但有些边界是直线，并不需要所有的点的坐标，只需要这条直线的两个端点而已，
此时参数设为cv2.CHAIN_APPROX_SIMPLE
'''
import cv2
import numpy as np
img=cv2.imread('black_white.jpg')
print(img.shape)
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
image = cv2.drawContours(img,contours,-1,(0,255,0),3)
cv2.imshow('result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
图像的矩可以帮助计算图像的质心、面积等
cv2.moments()函数计算的矩以字典的形式返回
根据矩可以求得对象的重心Cx=M10/M00,Cy=M01/M00
'''
import cv2
import numpy as np
img=cv2.imread('black_white.jpg')
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,1,2)
cnt = contours[0]
print(cnt[1][0])
M = cv2.moments(cnt)
print(M)
Cx = int(M['m10']/M['m00'])
Cy = int(M['m01']/M['m00'])

'''
轮廓面积
'''
area = cv2.contourArea(cnt)
print(area)
print(M['m00'])
#也可以使用0阶矩，M['m00']直接计算
'''
轮廓周长
'''
perimeter = cv2.arcLength(cnt,True) #封闭轮廓的周长
perimeter = cv2.arcLength(cnt,False) #曲线轮廓的周长


'''
轮廓近似：将轮廓形状近似到另一种由更少点组成的轮廓形状
新轮廓点的数目由我们设定的准确度来确定
函数是cv2.approxPolyDP()
'''
#epsilon是原始轮廓到近似轮廓的最大距离，很重要

import cv2
import numpy as np
img=cv2.imread('black_white.jpg')
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cnt = contours[0]
epsilon = 0.00001*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
print(approx)
image = cv2.drawContours(img,approx,-1,(0,255,0),2)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
凸包
凸包与轮廓近似相似，但不同，有些情况两者呈现的结果相同
cv2.convexHull()用来检测一个曲线是否具有凸性缺陷，并纠正缺陷
一般来说，凸性曲线总是凸出来的，至少是平的，如果有地方凹进去，则是凸性缺陷
参数：points:传入的轮廓
     hull：输出，一般不需要
     clockwise:方向标志，若为True,输出的凸包是顺时针
     returnPoints：默认值时True,返回凸包上点的坐标，若为False，返回与凸包点对应的轮廓上的点
'''
hull = cv2.convexHull(cnt,returnPoints=False)  #返回的是轮廓点的索引
print(hull)   

'''
凸性检测：检测一个曲线是不是凸的
'''
cv2.isContourConvex(cnt)


'''
边界矩形
直边界矩形，没有旋转的矩形，面积不是最小，
'''
#x,y为左上角顶点坐标，w,h是宽和高
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
旋转边界，考虑了对象的旋转，面积是最小的
'''
rect = cv2.minAreaRect(cnt) #获取矩形的中心点，长和宽,以及旋转角度
print(rect)
box = cv2.boxPoints(rect)  #获取矩形的四个角的坐标
box = np.int0(box)     #转换为整型
print(box)
image = cv2.drawContours(img,[box],0,(0,0,255),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
最小外界圆
'''
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)
cv2.imshow('circle',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''椭圆拟合'''
'''最后返回的结果其实就是旋转矩形的内切圆'''
ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(0,255,0),2)
cv2.imshow('ellipse',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#椭圆拟合的椭圆相关属性
(x,y),(MA,Ma),angle = cv2.fitEllipse(cnt)


'''直线拟合'''
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx)+y)
righty = int(((cols-x)*vy/vx)+y)
img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
cv2.imshow('line_fit',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''轮廓面积与凸包面积之比'''
area = cv2.contourArea(cnt)  #计算轮廓面积
hull = cv2.convexHull(cnt)  #找出凸包点
hull_area = cv2.contourArea(hull) #计算凸包面积
ratio = float(area)/hull_area
print(ratio)


'''
掩模与像素点
'''

mask = np.zeros(imgray.shape,np.uint8)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = cv2.drawContours(mask,[cnt],0,255,-1)
cv2.imshow('gray',imgray)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
a = np.ones((3,3))
print(a)
b = np.nonzero(a)
print(np.transpose(b))
'''

#使用掩模求得对象上最大值和最小值点以及它们的位置
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(imgray,mask=mask)
print(max_loc)


'''求对象的平均颜色及平均灰度'''
mean_val = cv2.mean(img,mask=mask)
print(mean_val)



'''
极点
一个对象最左边、最右边、最上边、最下面的点
'''

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
res = cv2.circle(image,bottommost,10,(0,0,255),-1)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
寻找凸缺陷
找一个点到多边形的最短距离
不同形状的匹配
'''
import cv2
import numpy as np

im = cv2.imread('star.jpg')
print(im.shape)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,240,255,cv2.THRESH_BINARY_INV)
contours,hierarchy = cv2.findContours(thresh,2,1)
print(len(contours))
print(len(contours[100]))
cnt = contours[100]
hull = cv2.convexHull(cnt,returnPoints=False)
defects = cv2.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far  =tuple(cnt[f][0])
    cv2.line(im,start,end,(0,255,0),2)
    
    cv2.circle(im,far,10,(0,255,0),-1)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
判断一个点到轮廓的最短距离，
当第三个参数为measureDist,设为False时，返回的是-1,0,1,内部为1，边界上为0，外面为-1
设为True时，计算实际的距离
'''
dist = cv2.pointPolygonTest(contours[10],(10,10),False)
print(dist)



'''
形状匹配
函数cv2.matchShapes()可以比较两个形状或者轮廓的相似度，如果返回值越小，匹配的越好
是根据Hu矩来计算的
Hu矩是归一化中心矩的线性组合，这样做是为了获取代表图像某个特征的矩函数
这些矩函数对某些变化，如缩放，旋转，镜像映射（除了h1）具有不变形
'''
import cv2
import numpy as np
im1 = cv2.imread('star.jpg',0)
im2 = cv2.imread('black_white.jpg',0)
ret1,thresh1 = cv2.threshold(im1,240,255,cv2.THRESH_BINARY_INV)
ret2,thresh2 =cv2.threshold(im2,127,255,0)
contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
contours2,hierarchy2 = cv2.findContours(thresh2,2,1)
cnt1 = contours1[100] 
cnt2 = contours1[0]
ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)







'''
轮廓的层次结构
在查找轮廓函数cv2.findContours(函数中，我们需要传入第二个参数轮廓检索模式
参数Contour_Retrieval_Mode，一般设为cv2.RETR_LIST或者cv2.RETR_TREE
在opencv中层次结构
每一个轮廓有都包含自己的信息，谁是父，谁是子等，用含有四个元素的数组表示[Next,Previous,First_Child,Parent]
Next表示同一级别的下一个轮廓
Previous：同一级别的上一个轮廓
First_Child:该轮廓的第一个子轮廓
Parent:该轮廓的父轮廓
如果上面对应的轮廓不存在，对应的值为-1
轮廓检索模式
cv2.RETR_LIST、cv2.RETR_TREE,cv2.RETR_CCOMP,cv2.RETR_EXTERNAL

cv2.RETR_LIST:提取所有轮廓，不创建任何父子关系
cv2.RETR_EXTERNAL：只返回最外边的轮廓，所有子轮廓被忽略
cv2.RETR_CCOMP：返回所有轮廓，但把每个轮廓只分为两个级别组织结构1，2，
一个对象的外轮廓为1，内部空洞轮廓为2
所以Previous=-1，Parent=-1
cv2.RETR_TREE：返回所有的轮廓，并且会构成一个完美的组织结构
'''











