# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:17:17 2019

@author: Administrator
"""

'''
使用opencv对图像进行傅里叶变换
使用Numpy中的FFT(快速傅里叶变换)函数
傅里叶变换的一些用处
函数：cv2.dft(),cv2.idft()
'''

'''
   原理：   使用2D离散傅里叶变换（DFT）分析图像的频域特性
            对于一个正弦信号，如果它的幅度变化非常快，我们可以说是高频信号，如果变化非常慢，称之为低频信号、
            应用到图像中：边界点和噪声是高频信号，没有如此大的幅度变化称之为低频信号
'''


##############################################################################
'''
使用Numpy实现傅里叶变换
函数：np.fft.fft2()
参数：1.输入图像，要求是灰度
     2.是可选的，决定输出数组大小，当大于输入图像大小时，在进行fft前，对输入图像进行补0
                                当小于输入图像时，输入图像就会被切割
得到的结果，频率为0的部分（直流分量）在输出图像的左上角，如果想让它在输出图像中心
则需要将结果沿两个方向平移N/2，函数np.fft.fftshift()可以实现
进行完频率变换之后，就可以构建振幅谱了
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg',0)
f = np.fft.fft2(img)
print(f.shape)
fshift = np.fft.fftshift(f)
#下面是构建振幅图的公式
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.title('Input image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('magnitude spectrum'),plt.xticks([]),plt.yticks([])
plt.show()

#上面的图像结果显示中心部分更白更亮，说明低频分量更多，主要信息位于中心部分

#上面操作过后，可以进行频域变换了，例如：高通滤波，重建图像（DFT的逆变换）
#操作：用一个60x60的矩形窗口对图像进行掩模操作从而去除低频分量，
#      然后使用函数np.fft.ifftshift()进行逆平移操作，直流分量将回到左上角
#      再使用函数np.ifft2()进行FFT逆变换，得到一堆复杂的数字，取绝对值

rows,cols = img.shape
crow,ccol = rows//2,cols//2
fshift[crow-30:crow+30,ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
#取绝对值
img_back = np.abs(img_back)

plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.title('Input image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_back)
plt.title('img_back'),plt.xticks([]),plt.yticks([])
plt.show()


######################################################################################




#####################################################################################

'''
opencv中实现傅里叶变换
函数：cv2.dft()和cv2.idft()
输出结果为双通道
第一通道是实数部分，第二通道是虚数部分
输入图像首先要转换成np.float32
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg',0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.title('Input image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('magnitude spectrum'),plt.xticks([]),plt.yticks([])
plt.show()


#前面使用了高通滤波器实现的边缘检测，现在使用低通滤波器LPH将高频去除，实现模糊化
#构建一个掩模，与低频区域对应的地方设置为1，与高频区域对应的地方设置为0

rows,cols = img.shape
crow,ccol = rows//2,cols//2
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 1

#运用mask和傅里叶反变换
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.title('Input image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_back,cmap='gray')
plt.title('img_back'),plt.xticks([]),plt.yticks([])
plt.show()

##################################################################################


'''opencv中关于傅里叶变换的函数要比numpy中的快，但是numpy函数对用户更加友好'''


'''
DFT的速度与输入图像的尺寸有关系
opencv中提供了获取最佳尺寸做DFT的函数cv2.getOptimalDFTSize()
'''
r,c = img.shape
br = cv2.getOptimalDFTSize(r)
bc = cv2.getOptimalDFTSize(c)
print(r,c)
print(br,bc)

# (342,548) ------(360,576)
#补0
''''
right = bc - c
bottom = br - r
bordertype = cv2.BORDER_CONSTANT
BEST_img = cv2.copyMakeBorder(img,0,bottom,0,right,bordertype,0)
'''
BEST_img = np.zeros((br,bc),np.uint8)
BEST_img[:r,:c] = img

cv2.imshow('BEST',BEST_img)
cv2.waitKey(0)
cv2.destroyAllWindows()






































