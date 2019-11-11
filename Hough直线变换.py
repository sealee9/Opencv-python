# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:49:56 2019

@author: Administrator
"""

'''
理解霍夫变换的概念
如何在图像中检测直线
函数：cv2.HoughLines(),cv2.HoughLinesP()

原理：    霍夫变换在检测各种形状的技术中非常流行，如果要检测的形状可以用数学表达式写出，就可以使用
         霍夫变换检测，即使要检测的形状存在一点破坏或者扭曲也可以使用

       一条直线可以用 y = Kx + c 或者 r = x COSθ + y SINθ 表示
            r是原点到直线的垂直距离，θ是直线的垂线与横轴顺时针的夹角

霍夫变换工作原理 ： 
                  每一条直线都可以用（r, θ）表示，所以首先创建一个2D数组（累加器），初始化这个累加器
                  所有的值为0，行表示r，列表示θ，这个数组的大小决定了最后结果的准确性。
                  如果需要角度的精度为1度，那就需要180列，对于r，最大值为图片对角线距离，如果精确度
                  要达到一个像素级别，那么行数应该与图像对角线的距离相等

现在如果我们有一个100x100的直线位于图像的中央，取直线上第一个点，知道了(x,y),把这个坐标
代入上面的公式，遍历θ的取值：0，1，2，3.....180，分别求出r，这样就有一系列的(r, θ)数值对
如果这个数值对在累加器中也存在相应的位置，就在这个位置加1，接下来再取之间第二个点，
重复操作后，取累加器中最大值的位置，这个位置(r, θ)就是一条直线

opencv中实现Hough变换

函数cv2.HoughLines(),返回值就是(r, θ)，r的单位是像素，θ的单位是弧度

函数参数：1.二值化图像，所以进行霍夫变换前首先进行二值化或者进行Canny边缘检测
         2，3参数分别代表r，θ的精度
         4.阈值，只有累加的值高于这个阈值时才被认为是一条直线
'''

import numpy as np
import cv2

img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

lines = cv2.HoughLines(edges,1,np.pi/180,200)

for rho,theta in lines.squeeze():
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
霍夫变换仅仅一条直线就需要遍历许多点才能得到参数
从图像上随机选取点来检测对于检测直线来说已经足够，同时对应得降低阈值，因为总的点数少了
Probabilistic_Hough_Transform就是对霍夫变换进行的优化
函数 cv2.HoughLinesP()
新增参数： 1.minLineLength  线的最短长度，比这个长度小的线被忽略
      2.MaxLineGap    两条线之间最大间隔，如果小于这值，两条直线被看成一条直线
这个函数的返回值就是直线的起点和终点。

'''


import numpy as np
import cv2

img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

minLineLength = 200
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#print(lines.shape)

for x1,y1,x2,y2 in lines.squeeze():
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





'''
Hough圆形变换
利用霍夫变换在图像中找出圆形
函数：cv2.HoughCircles()

原理：  圆形的数学表达式为(X-Xcenter)**2 + (Y - Ycenter) = R**2
       由上式看出，需要三个参数确定一个圆形
       因此使用霍夫变换的累加器必须是3维的
       这样效率会很低，所以opencv使用了霍夫梯度法，它可以使用边界的梯度信息

'''

import cv2
import numpy as np

img = cv2.imread('coins.jpg')
img = cv2.medianBlur(img,5)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#参数：1.输入图像，2.边缘检测方法 3.累加器分辨率，1代表与输入一致，2代表是输入的一半
 #      4.圆心之间的最小距离，5.Canny梯度检测的阈值 6.累加器的阈值 最后两个为圆的最大最小半径
 #主要调参数4，5，6
circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,10,param1=70,param2=100,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles.squeeze():
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2) #绘制找到的圆形
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),2) #绘制找到的圆心

cv2.imshow('Circle',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






























