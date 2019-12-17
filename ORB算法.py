# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:36:52 2019

@author: Administrator
"""
'''
BRIEF算法介绍(Binary Robust Independent Elementary Features)
由于SIFT算法使用128维的描述符，使用的是浮点数，所以要使用512个字节，同样SURF算法最少使用256个字节
创建一个包含上千个特征的向量需要消耗大量的内存，在嵌入式等资源有限的设备上是不合适的，匹配时还会消耗
更多的内存和时间
在实际匹配中其实并不需要如此多维度，我们可以使用PCA，LDA等方法来进行降维，甚至可以使用LSH(局部敏感哈希)
将SIFT浮点数的描述符转换成二进制字符串，对这些字符串再使用汉明距离进行匹配，汉明距离的计算只需要进行XOR
位运算以及位计数，这种计算适合现代cpu上进行，但是要先找到描述符
BRIEF就是解决这个问题的，他不去计算描述符而是直接找到一个二进制字符串，这种算法使用的是已经平滑后的图像，
它会按照一种特定的方式选取一组像素点对nd(x,y),然后在这些像素点对之间进行灰度值对比，例如：第一个点对的
灰度值分别为p,q,如果p小于q，结果就是1，否则就是0，这样就对nd个点对进行对比得到一个nd维的二进制字符串
nd可以是128，256，512，默认情况下256，当我们获得这些二进制字符串后就可以使用汉明距离对它们进行匹配了

非常重要一点：
              BRIEF是一种特征描述符，它不提供查找特征的方法，所以需要使用其他特征检测器，比如SIFT,SURF等
              文献推荐使用CenSurE特征检测器，该算法很快，BRIEF对CenSurE关键点的描述效果比SURF好
BRIEF是一种对特征点描述符计算和匹配的快速方法，这种算法可以实现很高的识别率，除非平面内的大旋转

'''
#这个算法代码在这个版本cv中不存在
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../Opencv/images/blox.jpg')

star = cv2.FeatureDetector_create('STARS')
brief = cv2.DescriptorExtractor_create('BRIEF')
sift = cv2.xfeatures2d.SIFT_create()
kp = star.detect(img,None)
kp,des = brief.compute(img,kp)

print(brief.getInt('bytes'))
print(des.shape)



'''
ORB(Oriented FAST and Rotated BRIEF)
SIFT和SURF需要算法专利保护，但ORB不需要
ORB基本是FAST关键点检测和BRIEF关键点描述器的结合体
首先它使用FAST找到关键点，然后再使用Harris角点检测对这些关键点进行排序找到其中的前N个点，
也使用金字塔从而产生尺度不变形特征，但存在是的一个问题是：FAST算法不计算方向，那旋转不变形怎么解决？
作者做了如下修改：  使用灰度矩的算法计算出角点的方向，以角点到角点所在区域质心的方向为向量的方向，
进一步提高旋转不变性，要计算以角点为中心半径为r的圆形区域的矩，再根据矩计算出方向
对于描述符，ORB使用的是BRIEF描述符，但由于BRIEF对于旋转是不稳定的，所以在生成特征前，要把关键点领域
的这个patch的坐标轴旋转到关键点的方向

实验证明，BRIEF算法的每一位的均值接近0.5，并且方差很大，steered_BRIEF算法的每一位的均值比较分散，这
导致方差减小，数据的方差大的一个好处是：使得特征更容易分辨，为了对steered_BRIEF算法使得特征的方差减小
的弥补和减少数据间的相关性，用一个学习算法(learning method)选择二进制测试的一个子集

文章中说ORB比DIFT和SURF快很多，是低功耗设备的最佳选择

使用函数cv2.ORB()创建一个ORB对象
参数：nfeature(是最有用的参数):默认值是500，表示要保留特征的最大数目
     scoreType设置使用Harris打分还是使用FAST打分对特征进行排序，默认是Harris打分
     WTA_K决定了产生每个oriented_BRIEF描述符要使用的像素点数目，默认值是2，也就是一次选择两个点
     在这种情况下进行匹配，要使用NORM_HAMMING距离
     如果WTA_K被设置成3或者4，那匹配距离就要设置为NORM_HAMMING2
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('../Opencv/images/blox.jpg')

orb = cv2.ORB_create()
kp = orb.detect(img,None)
kp,des = orb.compute(img,kp)
img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
plt.imshow(img2),plt.show()