# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:15:43 2019

@author: Administrator
"""

'''
简单阈值，自适应阈值，Otsu's二值化等
学习的函数：cv2.threshold,cv2.adaptiveThreshold等
'''

'''全局阈值，用一个阈值在全图上做变换'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)    
#参数：被处理图像，阈值，更新阈值，阈值方法
ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)  #二值化，高于阈值部分用更新阈值替代
ret,th2 = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV) #低于阈值部分用更新阈值取代
ret,th3 = cv2.threshold(img,120,255,cv2.THRESH_TRUNC)  #截断阈值化
ret,th4 = cv2.threshold(img,120,255,cv2.THRESH_TOZERO)  #低于阈值部分取0
ret,th5 = cv2.threshold(img,120,255,cv2.THRESH_TOZERO_INV)  #高于阈值部分取0
title = ['img','th1','th2','th3','th4','th5']
image = [img,th1,th2,th3,th4,th5]
for i in range(len(title)):
    plt.subplot(2,3,i+1),plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()



'''
自适应阈值，在图像不同区域采用不同的阈值
方法：1.cv2.ADAPTIVE_THRESH_MEAN_C，阈值取自领域区域的平均值
2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,阈值取自领域区域的加权和，权重为一个高斯窗口

BLOCK_SIZE：领域大小
C：常数，最后的阈值是上述方法得到后的值减去C
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
#img = img.astype(np.float32)
img = cv2.medianBlur(img,5)    
ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY) 
#参数：被处理图像，更新阈值，阈值选取方法，阈值方法，领域大小， 
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)  
title = ['img','GLOBAL','ADAPTIVE_MEAN','ADAPTIVE_GAUSSIAN']
image = [img,th1,th2,th3]
'''
for i in range(len(title)):
    cv2.imshow(title[i],image[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
for i in range(len(title)):
    plt.subplot(2,2,i+1),plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()



'''
Otsu's二值化
在全局阈值中，我们要不断尝试选取合适的阈值对图像进行阈值化处理
对于图像直方图是双峰时，使用Otsu二值化可以自动寻找最佳的阈值
这对于双峰图像有优势，对于非双峰图像可能不理想
使用的函数是cv2.threshold(),要额外增加一个参数cv2.THRESE_OTSU,
最佳阈值的返回值是retVal，使用该二值化时，函数内的阈值要设为0
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
#img = img.astype(np.float32)
img = cv2.medianBlur(img,5)    
ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY) 
print(ret)
ret,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
title = ['img','GLOBAL','OTSU']
image = [img,th1,th2]
for i in range(len(title)):
    plt.subplot(3,1,i+1),plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()









