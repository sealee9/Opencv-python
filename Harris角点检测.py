# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:45:34 2019

@author: Administrator
"""

'''
理解Harris角点检测的概念
函数：cv2.cornerHarris(),cv2.cornerSubPix()

原理：    角点有一个特性：向任何方向移动变化都很大
         
         将窗口向各个方向移动(u,v)，然后计算所有差异的总和，公式如下
         E(u,v) = ∑ W(x,y) [I(x+u,y+v)-I(x,y)]**2
         W是窗口函数，可以是正常的矩形窗口也可以是对每一个像素给予不同权重的高斯窗口
         角点检测中要使E(u,v)最大，所以必须使方程右边的差的平方取值最大
         
Opencv中函数cv2.cornerHarris()用来角点检测
参数： 1. img，数据类型为float32的输入图像
      2. blockSize  角点检测中要考虑的领域大小
      3. Ksize ：Sobel求导中使用的窗口大小
      4. K       Harris角点检测方程中的自由参数，取值参数为[0.04,0.06]      
         
'''
import cv2
import numpy as np

img = cv2.imread('images/blox.jpg')

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray = np.float32(img_gray)

dst = cv2.cornerHarris(img_gray,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst>0.01*dst.max()] = [0,0,255]

cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
亚像素级精确度的角点

有时需要最大精度的角点检测，opencv提供了cv2.cornerSubPix()函数，可以实现亚像素级别的角点检测

步骤： 1. 找到Harris角点
      2. 将角点的重心传给这个函数进行修正
Harris角点用红色像素标出，绿色像素是修正后的像素，
在使用这个函数时要定义一个迭代停止条件，当迭代次数达到或精度条件满足迭代就会停止。
同样需要定义进行角点搜索的领域大小
'''

import cv2
import numpy as np

img = cv2.imread('images/blox.jpg')

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray = np.float32(img_gray)

dst = cv2.cornerHarris(img_gray,2,3,0.04)

dst = cv2.dilate(dst,None)
ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#找到重心
ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)

#定义迭代停止条件和修正角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.001)
corners = cv2.cornerSubPix(img_gray,np.float32(centroids),(5,5),(-1,-1),criteria)
res = np.hstack((centroids,corners))
#省略小数点后面的数字
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow('subpixel',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''
Shi-Tomasi 角点检测&适合于跟踪的图像特征
函数：cv2.goodFeatureToTrack()

Harris角点检测的打分公式为：     R = a1a2-k(a1+a1)**2
Shi-Tomasi角点检测的打分公式为： R = min(a1,a2)

如果打分超过了阈值，我们就认为它是一个角点，就可以把它绘制到a1~a2空间中，只有a1,a1都大于最小值
才被认为是角点


Opencv中提供了函数：cv2.goodFeatureToTrack()可以用来实现Shi-Tomasi角点检测
这个方法可以获取图像中N个最好的角点，通常情况下输入的是灰度图像，然后确定要检测的角点数目
再设置角点的质量水平，0-1之间，它代表角点的最低质量，低于这个数的角点会被忽略，最后设置两个角点之间
最短欧氏距离

找到所有角点后，将所有低于质量水平的角点忽略
然后把合格的角点按角点质量进行降序排列
函数会用质量最好的那个角点，然后将它最小距离的角点删掉，按照这样的方式放回N个最佳角点
'''

import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/blox.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#餐胡：1.输入图像，需要的角点个数，角点最低质量，两个角点之间的最短欧氏距离
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10) #返回的是[[]]两层括号的数组
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


cv2.imshow('Shi-Tomasi',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





















































