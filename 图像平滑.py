# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:28:22 2019

@author: Administrator
"""

'''
使用不同的低通滤波器对图像进行模糊
使用自定义滤波器对图像进行卷积（2D卷积）

低通滤波器（LFP）帮助去除图像噪声和模糊图像
高通滤波器（HFP）找到图像的边缘
'''

'''使用卷积核对图像进行卷积，实现平均滤波'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 

img = cv2.imread('cat.jpg')
def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
img = sp_noise(img,0.1)
kernel = np.ones((5,5),np.float32)/25
#第二个参数-1表示输出图像的深度和输入图像保持相同
change_img = cv2.filter2D(img,-1,kernel)
plt.subplot(1,2,1),plt.imshow(img),plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2),plt.imshow(change_img),plt.title('Averiging')
plt.xticks([])
plt.yticks([])
plt.show()


'''
图像模糊（平滑）
opencv提供了四种模糊方法
'''

'''平均，使用归一化卷积框用cv2.blur(),不使用归一化卷积框：cv2.boxFilter(),这时要传入参数normalize=False'''
blur = cv2.blur(img,(5,5))
plt.subplot(1,2,1),plt.imshow(img),plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2),plt.imshow(blur),plt.title('Averiging')
plt.xticks([])
plt.yticks([])
plt.show()

blur = cv2.boxFilter(img,-1,(5,5),normalize=False)
plt.subplot(1,2,1),plt.imshow(img),plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2),plt.imshow(blur),plt.title('Averiging')
plt.xticks([])
plt.yticks([])
plt.show()

'''
高斯模糊
将卷积核换成高斯核，原来的核中每个值相等，现在核中中心点的值最大，其余值根据距离中心点的距离递减
构成一个高斯小山包，高斯核的宽高必须是奇数，实现函数：cv2.GaussianBlur()，
参数有高斯函数X，Y方向的标准差，若只规定X的标准差，Y与之保持一致，
若两者都是0，高斯函数会根据核函数的大小自己计算
cv2.getGaussianKernel()函数用来构建高斯核
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 

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


image = cv2.imread('cat.jpg')
image_new = gasuss_noise(image,0.1)
cv2.imshow('image',image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
blur = cv2.GaussianBlur(image_new,(3,3),0)
cv2.imshow('image1',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''中值滤波'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 

img = cv2.imread('cat.jpg')
def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
img = sp_noise(img,0.25)
res = cv2.medianBlur(img,5)
cv2.imshow('original',img)
cv2.imshow('res',res)
cv2.waitKey()
cv2.destroyAllWindows()


'''双边滤波'''

'''
双边滤波函数cv2.bilateralFilter()能在保持边界清晰的情况下有效的去除噪音，但这种操作要比其他滤波器慢，
高斯滤波器是求中心点邻近区域像素的高斯加权平均值，它只考虑了像素之间的空间关系，而不会考虑像素之间的相似性
例如灰度值关系
这种方法不会考虑一个像素是否位于边界上，所以边界被模糊了

双边滤波同时使用空间高斯权重和灰度值相似性高斯权重，灰度值相似性高斯函数确保只与中心点像素灰度值相近的才会
被用来做模糊运算，这种方法确保边界不会被模糊掉，因为边界处的灰度值变化大
'''

#9是领域直径，后面两个参数分别是：空间高斯函数的标准差，灰度值相似性高斯函数标准差
blur = cv2.bilateralFilter(img,9,75,75)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 

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


image = cv2.imread('cat.jpg')
image_new = gasuss_noise(image,0.1)
cv2.imshow('image',image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
blur = cv2.bilateralFilter(image_new,9,75,75)
cv2.imshow('image1',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()











