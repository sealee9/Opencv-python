# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:45:37 2019

@author: Administrator
"""

'''
使用SVM对手写数据进行分类
在knn中是直接使用像素的灰度值作为特征向量，这次使用的是方向梯度直方图HOG作为特征向量

'''
import cv2
import numpy as np

SZ = 20
bin_n = 16
#svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,svm_type = cv2.ml.SVM_C_SVC,
 #                 C=2.67,gamma=5.383)
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
#抗扭斜函数
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02'])<1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1,skew,-0.5*SZ*skew],[0,1,0]])
    img = cv2.warpAffine(img,M,(SZ,SZ),flags=affine_flags)
    return img

#计算图像的HOG描述符，创建一个函数HOG，为此要计算图像X方向和Y方向的Sobel导数，然后
#计算得到每个像素的梯度的方向和大小，把这个梯度转换成16位的整数，将图像分为4个小的方块
#对每个小方块计算它们的朝向直方图(16个bin)，使用梯度的大小做权重，这样每个小方块都会得到
#一个含有16个成员的向量，4个小方块的4个向量就组成了这个图像的特征向量(包含64个成员),
#这就是我们要训练数据的特征向量
def hog(img):
    gx = cv2.Sobel(img,cv2.CV_32F,1,0)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1)
    mag,ang = cv2.cartToPolar(gx,gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:10,:10],bins[10:,:10],bins[:10,10:],bins[10:,10:]
    mag_cells = mag[:10,:10],mag[10:,:10],mag[:10,10:],mag[10:,10:]
    hists = [np.bincount(b.ravel(),m.ravel(),bin_n) for b,m in zip(bin_cells,mag_cells)]
    hist = np.hstack(hists)
    return hist

img = cv2.imread('digits.png',0)

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

#训练

deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]


traindata = np.float32(hogdata).reshape(-1,64)

responses = np.repeat(np.arange(10),250)[:,np.newaxis]

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(traindata,cv2.ml.ROW_SAMPLE,responses)
svm.save('svm_data.dat')

#testing

deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]

testdata = np.float32(hogdata).reshape(-1,bin_n*4)

result = svm.predict(testdata)[1]
mask = result==responses
print(result)
correct = np.count_nonzero(mask)
accuracy = correct*100.0/result.size
print(accuracy)
















    
    
    
    






