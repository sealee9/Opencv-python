# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:01:05 2019

@author: Administrator
"""

'''
使用非局部平均值去噪算法去除图像中的噪音
函数有：cv2.fastNIMeansDenoising(),cv2.fastNIMeanDenoisingColored()等
原理：
        在前面的技术手段中，有高斯平滑、中值平滑等，当噪声比较小时，这些技术的效果都是很好的，在这些技术中
        选取的是像素周围一个小的领域然后用高斯平局值或者中值平均值取代中心像素，简单来说，像素级别的噪声的
        去除是限制在局部领域的。
        噪声有一个性质，认为噪声是平均值唯一的随机变量，考虑一个带噪声的像素点，p=p0+n,其中p0是像素的真实
        值，n为这个像素的噪声，我们可以从不同图片中选取大量的相同像素然后计算平均值，理想情况下，我们会得到
        p=p0，因为像素的平均值为0。
        通过简单的设置就可以去除这些噪声，将一个静态摄像头固定在一个位置连续拍摄几秒钟，这样就会得到足够多的
        图像帧，或者同一场景的大量图像，求解这些帧的平均值，将最终结果与第一帧图像对比一下，会发现噪声减小了
        但是这种简单的方法对于摄像头和运动场景并不适用，大多数情况下我们只有一张带有噪声的图像。
    想法很简单，需要一组相似的图片，通过取平均值的方法可以去除噪声，考虑图像中一个小的窗口(5x5),有很大可能
    图像中的其他区域也存在一个相似的窗口，有时这个相似窗口就在领域周围，如果我们找到这些相似的窗口并取它们的
    平均值会怎么样呢？
    操作：我们可以选取包含目标像素的一个小窗口，然后在图像中搜索相似的窗口，最后求取所有窗口的平均值，并用
    这个值取代目标像素的值，这种方法就是非局部平均值去噪，跟之前的平滑技术相比，这种算法要消耗更多的时间，
    但是结果很好。
    
opencv中的图像去噪：opencv中提供了这种技术的四个变本
                    1.cv2.fastNIMeansDenoising() 使用对象为灰度图
                    2.cv2.fastNIMeansDenoisingColored() 使用对象为彩色图
                    3.cv2.fastNIMeansDenoisingMulti() 适用于短时间的图像序列（灰度图像）
                    4.cv2.fastNIMeansDenoisingColoredMulti() 适用短时间的图像序列（彩色图像）
                共同参数：
                        1、 h：决定过滤器强度，h值很高可以很好的去除噪声，但也会把图像的细节抹去(取10效果不错)
                        2、hForColorComponents:与h相同，但适用于彩色图像(与h相同)
                        3、tempalteWindowSize():奇数，（推荐值为7）
                        4、searchWindowSize():奇数，(推荐值为21)                                               
                        
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

def gasuss_noise(image, mean=0, var=0.01):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out
image = cv2.imread('../Opencv/images/home.jpg')
image_new = gasuss_noise(image,0.1)
cv2.imshow('image',image_new)

dst = cv2.fastNlMeansDenoisingColored(image_new,None,10,10,7,61)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


##############################################################################
##############################################################################
'''
对视频进行操作

第一个参数：带有噪声的图像帧列表
第二个参数：imgtoDenoiseIndex 设定哪些帧需要去噪，可以传入一个帧的索引
第三个参数：temporaWindowSize，可以设置用于去噪的相邻帧的数目，应该是一个奇数
            例如传入5帧图像，imgToDenoiseIndex=2和temporalWindowSize()=3，那么表示
            第一帧、第二帧、第三帧图像将被用于第二帧图像进行去噪
'''

import cv2
import numpy as np 
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('../Opencv/images/vtest.avi')

img = [cap.read()[1] for i in range(5)]

#转换成灰度图像
gray = [cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in img]


gray_ = [np.float64(i) for i in gray]

noise = np.random.randn(*gray[1].shape)*5

noisy = [i+noise for i in gray_]

noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

dst = cv2.fastNlMeansDenoisingMulti(noisy,2,5,None,4,7,35)
#plt.subplot(131),plt.imshow(gray[2],'gray')
#plt.subplot(132),plt.imshow(noisy[2],'gray')
#plt.subplot(133),plt.imshow(dst,'gray')
cv2.namedWindow('origin',cv2.WINDOW_NORMAL)
cv2.imshow('origin',gray[2])
cv2.namedWindow('noise',cv2.WINDOW_NORMAL)
cv2.imshow('noise',noisy[2])
cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
































