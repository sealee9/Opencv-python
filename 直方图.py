# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:37:43 2019

@author: Administrator
"""

'''
使用opencv或者numpy计算直方图
使用opencv或者matplotlib绘制直方图
函数有：cv2.calcHist(),  np.histogram()

直方图：对整幅图像的灰度分布有一个整体的了解，对图像的对比度，亮度，灰度分布有一个直观地认识

统计直方图：   BINS：如果不需要知道每个像素值的像素点数目，而只需要知道两个像素值之间的像素点数目，设置该参数
              DIMS：收集的参数数目，我们只考虑灰度值的话，设为1
              RANGE：统计的灰度值范围，一般来说为[0,256]
'''


'''
使用opencv统计直方图
函数：cv2.calcHist(images,channels,mask,histSize,ranges)
images:原图像(格式是uint8或者float32)，当传入函数时，用中括号[]，比如[image]
channels:同样需要中括号[],如果输入的灰度图像，则为[0],如果是彩色图像，则为[0],[1],[2],对应着B/G/R
mask:掩模图像，如果统计的是整幅图像，则为None,要是统计图像中一部分的话，要制作一个掩模，然后使用掩模
histSize:BIN的数目，需要中括号[],如：[256]
ranges:像素范围，[0,256]
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
print(hist.shape)
plt.plot(hist)
plt.show()

'''使用numpy统计直方图'''
#img.ravel()将图像转为一维数组
hist,bins = np.histogram(img.ravel(),256,[0,256])
print(hist.shape)   #bins=257,因为该方法中是0-0.99，最后加上一个256，所以为257

plt.plot(hist)
plt.show()
#还有一个函数np.bincount(),比上面的速度快
hist = np.bincount(img.ravel(),minlength=256)
print(hist)

'''注意：opencv函数要比np.histogram()快40倍'''


'''
绘制直方图
上面先使用函数统计直方图，在绘制
也可以直接统计并绘制，如下
'''
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg',0)
plt.hist(image.ravel(),256,[0,256])
plt.show()

'''绘制多通道的直方图'''
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg')
color =('b','g','r')
for i,c in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=c)
plt.show()

'''也可以先使用split函数分离通道再绘制各个通道的灰度值直方图，此函数比较慢'''
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg')
b,g,r = cv2.split(image)
hist = np.bincount(b.ravel(),minlength=256)
print(hist.shape)
plt.plot(hist)
plt.show()


'''
如果只需要统计图像中部分区域的直方图
使用掩模
然后将掩模传递给统计函数
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg',0)
mask = np.zeros(img.shape[:2],np.uint8)
mask[100:600,100:600] = 255
mask_img = cv2.bitwise_and(img,img,mask=mask)
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(2,3,1),plt.imshow(img,'gray')
plt.subplot(2,3,2),plt.imshow(mask,'gray')
plt.subplot(2,3,3),plt.imshow(mask_img,'gray')
plt.show()
plt.plot(hist_full,color='r'),plt.plot(hist_mask,color='g')
plt.show()




'''直方图均衡化'''
'''opencv'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('aero.jpg',0)
img_equal = cv2.equalizeHist(img)
res = np.hstack((img,img_equal))
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(img.flatten(),256)
plt.show()
plt.hist(img_equal.flatten(),256)
plt.show()


'''numpy实现均衡化'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('aero.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
#计算累计分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized,color='b')
plt.hist(img.flatten(),256,[0,256],color='r')
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()
#构建掩模数组，cdf为原数组，当数组元素为0 时，掩盖（计算时被忽略）
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())
#对被掩盖的元素赋值，这里赋值0
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
cv2.imshow('image1',img)
cv2.imshow('image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(img.ravel(),256)
plt.hist(img2.ravel(),256)
plt.show()



'''
CLAHE有限对比适应性直方图均衡化
由于直接直方图均衡化对图像的灰度值集中在某一范围比较有效
但是 如果图像的灰度值并不是单纯的集中在某一灰度值范围
在上面情况下使用直方图均衡化会损失一些信息
所以要使用自适应的直方图均衡化
把图像分成很多小块，对每个小块进行均衡化
代码如下：
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('orange.jpg',0)

img_equal = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img_adp_equal = clahe.apply(img)
cv2.imshow('original',img)
cv2.imshow('original_equal',img_equal)
cv2.imshow('original_adp_equal',img_adp_equal)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(img.ravel(),256,color='r')
plt.hist(img_equal.ravel(),256,color='g')
plt.hist(img_adp_equal.ravel(),256,color='b')
plt.show()

'''
调整图像的对比度和亮度
原理：原图像像素乘以alpha再加上beta
alpha一般取255/img.max()增加亮度
'''

res = cv2.convertScaleAbs(img,alpha=1.5,beta=10)
cv2.imshow('original',img)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()





















