# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:30:21 2019

@author: Administrator
"""

'''
一系列不同分辨率的原图像
最大的图像放在下面，最小的放在上面，类似于金字塔结构
顶部图像是通过将底部图像中连续的列和行去除得到的
顶部图像中每个像素等于下一层5个像素的高斯加权平均值
这样操作一次MXN图像，变为M/2 x N/2大小的图像，大小逐渐减小
函数：cv2.pyrDown()和cv2.pyrUp()

'''
import cv2
img = cv2.imread('cat.jpg')
print(img.shape)
lower_img = cv2.pyrUp(img)
print(lower_img.shape)
cv2.imshow('Lower',lower_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
图像融合
如果直接将两幅图像直接拼接在一起，会出现不连续
使用图像金字塔融合能解决
步骤：
1.读入两幅图片
2.分别构建两幅图的图像金字塔（6层）
3.根据高斯金字塔计算拉普拉斯金字塔
4.在拉普拉斯的每一层进行融合，图像的左边和另一个图像的右边
5.根据融合后的图像金字塔重建原图像

拉普拉斯金字塔
Li = Gi - cv2.pyrUp(Gi+1)
Gi+1 = cv2.pyrDown(Gi)

'''

import cv2
import numpy as np
A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
    
B = B.copy()
gpB = [B]
for i in range(6):
    B = cv2.pyrDown(B)
    gpB.append(B)

    
laA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    laA.append(L)


laB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    laB.append(L)
LS = []
for la,lb in zip(laA,laB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:int(cols/2)],lb[:,int(cols/2):]))
    LS.append(ls)

    
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_,LS[i])
    
real = np.hstack((A[:,0:int(cols/2)],B[:,int(cols/2):]))
cv2.imwrite('pyramid_bleding2.jpg',ls_)
cv2.imwrite('direct_bleding2.jpg',real)




















