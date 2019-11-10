# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:10:23 2019

@author: Administrator
"""

'''
使用模板匹配在一幅图像中查找目标
函数：cv2.matchTemplate(),cv2.minMaxLoc()

原理：    和2D卷积一样，也是用模板图像在输入图像（大图）上滑动，并在每一个位置对模板图像和与其对应的
          的输入图像的子区域进行比较，返回的结果是一个灰度图像，每一个像素表示了此区域与模板的匹配程度
          opencv提供了几种不同的比较方法

         如果输入图像的大小是 WxH,模板的大小是wxh，输出图像的大小就是W-w+1,H-h+1,然后使用
         函数cv2.minMaxLoc()来找其中的最大值和最小值的位置，第一个值为矩形左上角的点，
         （w,h）为模板矩形的宽和高，这个矩形就是找到的模板区域了
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

origin = cv2.imread('messi.jpg')
img = cv2.imread('messi.jpg',0)
img1 = img.copy()

im_s = cv2.imread('messi_face.jpg',0)
#由于读取的是高河宽。所以要改成宽和高
w,h = im_s.shape[::-1]

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img1.copy()
    method = eval(meth)
    #使用模板匹配
    res = cv2.matchTemplate(img,im_s,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    print(min_val,max_val)
    print(min_loc,max_loc)
    #如果使用的方法是TM_SQDIFF
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w,top_left[1] + h)
    
    cv2.rectangle(img,top_left,bottom_right,255,2)
    
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.title('Matching Result'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap='gray')
    plt.title('Detected Point'),plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    plt.show()

cv2.imshow('Face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('persons.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

target = cv2.imread('face_one.jpg',0)

w,h = target.shape[::-1]

res = cv2.matchTemplate(img_gray,target,cv2.TM_CCOEFF_NORMED)
threshold = 0.32
loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,0),1)
    
cv2.namedWindow('Face',cv2.WINDOW_NORMAL)
cv2.imshow('Face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
















































