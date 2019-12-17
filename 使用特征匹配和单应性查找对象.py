# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:16:17 2019

@author: Administrator
"""

'''
目标：   联合使用特征提取和calib3d模块中的findHomography在复杂图像中查找已知对象

在特征匹配中，我们使用一个查询图像，在其中找到一些特征点(关键点)，又在另一幅图像中也找到了一些特征点，最后
对这两幅图像之间的特征点进行匹配，简单来说：在一张杂乱的图像中找到一个对象的某些部分的位置，这些信息足以
帮助我们在目标图像中准确地找到查询对象。

为了达到上述目的，可以使用calib3d模块中的cv2.findHomography()函数，如果将这两幅图像中的特征点集传给这个
函数，函数就会找到这个对象的透视图变换，然后就可以使用函数cv2.perspectiveTransform()找到这个对象了，
至少要4个正确的特征点才能找到这种变换。

在匹配过程中可能会有一些错误，而这些错误会影响最终结果，为了解决这个问题，算法使用RANSAC和LEAST_MEDIAN
所以好的匹配提供的正确的估计被称为inliers，剩下的被称为outliers，cv2.findHomography()返回一个掩模，
这个掩模确定了inlier和outlier点
'''

import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('../Opencv/images/box.png',0) 
img2 = cv2.imread('../Opencv/images/box_in_scene.png',0)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)
des1 = np.float32(des1)
des2 = np.float32(des2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
#我们设置只有存在10个以上匹配时才去查找目标(MIN_MATCH_COUNT=10)，否则显示警告信息：“现在匹配不足”
#如果找到了足够的匹配，我们要提取两幅图像中匹配点的坐标，把他们传入到函数中计算透视变换，一旦我们找到
#3x3的变化矩阵，就可以使用它将查询图像的四个顶点(四个角)变换到目标图像中去，然后再绘制出来
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    cv2.polylines(img2,[np.int32(dst)],True,255,10,cv2.LINE_AA)
    
else:
    print("Not enough matches are found - %d/%d"%(len(good),MIN_MATCH_COUNT))
    matchesMask = None 

#如果能成功找到目标图像的话，就绘制Inliers,如果失败，就绘制匹配的关键点
draw_params = dict(matchColor=(0,255,0),singlePointColor=None,matchesMask=matchesMask,flags=2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow('result',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
    
    
    
    
        
        
        





