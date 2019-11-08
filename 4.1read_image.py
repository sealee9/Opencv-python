# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:27:15 2019

@author: Administrator
"""



import numpy as np
import cv2
#读入图片，第一个参数是图片的路径
#第二个参数可以是：cv2.IMREAD_UNCHANGED 读入的图片包括alpha通道
#cv2.IMREAD_COLOR读入彩色图片，是默认值
#cv2.IMREAD_GRAYSCALE 以灰色图片形式读入
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)  #键盘绑定函数，等待键盘输入，里面参数是毫秒级别，0表示无限期等待键盘输入
#删除任何我们建立的窗口，删除特定窗口用：cv2.destroyWindow('窗口名')
cv2.destroyAllWindows()  

import numpy as np
import cv2
img = cv2.imread('cat.jpg',cv2.IMREAD_UNCHANGED)
#也可以提前创建一个窗口用来加载图片
#第二个参数为：cv2.WINDOW_AUTOSIZE窗口初始设定，不能改变窗口的大小
#为cv2.WINDOW_NORMAL时，可以调整窗口大小
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#############################################################
'''保存图像'''
cv2.imwrite('imggray.png',img)


'''加载一张图片并显示，按esc不保存退出，按s保存并退出'''
import numpy as np
import cv2
img = cv2.imread('cat.jpg')
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('cat_sub.jpg',img)
    cv2.destroyAllWindows()
    
'''使用matplotlib显示图片'''
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg')
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
'''注意：由于cv2加载图片的模式是BGR，而matplotlib的模式是RGB，所以两者联合在彩色上操作时会有异常'''


