# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:06:37 2019

@author: Administrator
"""

'''
学习在图像中进行特征匹配
使用opencv中蛮力(Brute-Force)匹配和FLANN匹配
'''

'''
Brute-Force匹配的基础：   首先在第一幅图像中选取一个关键点，然后依次与第二幅图像的每个关键点进行(描述符)距离测试
                         最后返回距离最近的关键点
  对于BF匹配器，首先使用cv2.BFMatcher()创建一个BF-marcher对象，有两个可选参数，第一个：normType，指定要使用的距离测试类型
默认值是cv2.Norm_L2,这种适合SIFT和SURF等，对于要使用二进制描述符的ORB,BRIEF,BRISK算法等，要使用cv2.NORM_HAMMING,
这样就返回两个测试对象之间的汉明距离，如果ORB算法的参数设置为VTA_K==3或者4，NormType就应该设置成cv2.NORM_HAMMING2
第二个参数：布尔变量crossCheck，默认值是False，如果设置成True，匹配条件就是更加严格，只有A中的第i个特征与B中的第j个特征点
距离最近，并且B中第j个特征点到A中的第i个特征点也是最近时才会返回最佳匹配(i,j),也就是这两个特征点要相互匹配才行

BFMatcher对象具有两个方法，BFMatcher.match()和BFMatcher.knnMatch()，
第一个方法会返回最佳匹配，第二个方法为每个关键点返回k可最佳匹配(降序排列之后取前k个)，其中k由用户设定
就像使用cv2.drawKeypoints()绘制关键点一样，可以使用cv2.drawMatches()来绘制匹配的点，它会将两幅图像先水平排列
然后在最佳匹配的点之间绘制直线(从原图像到目标图像)，如果前面使用的是BFMatcher.knnMatch(),可以使用函数成。drawMatchesKnn
为每个关键点和它的最佳匹配点绘制线
'''

'''对ORB描述符进行蛮力匹配'''
import numpy as np
import cv2

img1 = cv2.imread('../Opencv/images/box.png',0) 
img2 = cv2.imread('../Opencv/images/box_in_scene.png',0)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches,key=lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],img2,flags=2)

cv2.imshow('Match',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''使用BF的knn匹配'''
import numpy as np
import cv2

img1 = cv2.imread('../Opencv/images/box.png',0) 
img2 = cv2.imread('../Opencv/images/box_in_scene.png',0)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

matches = bf.knnMatch(des1,des2,k=1)

#matches = sorted(matches,key=lambda x:x.distance)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,img2,flags=2)

cv2.imshow('Match',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
FLANN匹配：    
               FLANN是快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbors)的简称
               是一个对大数据集和高维特征进行最近邻搜索的算法的集合，而且这些算法都已经被优化过了，
               在面对大数据集时它的效果要好于BFMatcher。
               使用FLANN，需要传入两个字典作为参数，这两个用来确定要使用的算法和其他相关信息等，
               第一个参数：IndexParams
               第二个参数：SearchParams，用来指定递归遍历的次数，值越高，结果越准确，但消耗的时间越多
'''
import numpy as np
import cv2

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

matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow('RESULT',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()











