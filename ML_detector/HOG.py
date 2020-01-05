# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:54:28 2020

@author: Administrator
"""

#first part

import cv2
import numpy as np

'''
读入彩色图像，并转换为灰度值图像, 获得图像的宽和高。采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）
目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音。采用的gamma值为0.5
'''
img = cv2.imread('images/messi.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('Image', img)
#cv2.imwrite("Image-test.jpg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img = np.sqrt(img / float(np.max(img)))
#cv2.imshow('Image', img)
#cv2.imwrite("Image-test2.jpg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#######################################################
'''
计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；求导操作不仅能够捕获轮廓，
人影和一些纹理信息，还能进一步弱化光照的影响。在求出输入图像中像素点（x,y）处的水平方向梯度、
垂直方向梯度和像素值，从而求出梯度幅值和方向
'''
# second part

height, width = img.shape
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
gradient_magnitude = cv2.magnitude(gradient_values_x,gradient_values_y)
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
print(gradient_magnitude.shape, gradient_angle.shape)
###############################################################################
'''
我们将图像分成若干个“单元格cell”，默认我们将cell设为8*8个像素。假设我们采用8个bin的直方图来
统计这6*6个像素的梯度信息。也就是将cell的梯度方向360度分成8个方向块，
例如：如果这个像素的梯度方向是0-22.5度，直方图第1个bin的计数就加一，
这样，对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），
就可以得到这个cell的梯度方向直方图了，就是该cell对应的8维特征向量而梯度大小作为投影的权值
'''
# third part

cell_size = 4
bin_size = 8
angle_unit = 360 / bin_size
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))

print(cell_gradient_vector.shape)

def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit)%8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        print(cell_angle.max())

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
###########################################################################
#将得到的每个cell的梯度方向直方图绘出，得到特征图
# fourth part

import math
import matplotlib.pyplot as plt

hog_image= np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

#plt.imshow(hog_image, cmap=plt.cm.gray)
#plt.show()
cv2.imshow('result',hog_image)
cv2.imshow('origin',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





















