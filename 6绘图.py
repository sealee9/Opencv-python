# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:58:39 2019

@author: Administrator
"""

import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
'''在图中画一条线，参数分别是：img(想要绘制的图片)起点，终点，颜色，线的宽度'''
cv2.line(img,(0,0),(200,200),(255,0,0),5)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''绘制矩形'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''绘制圆形'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
'''参数为要绘制的图形，圆心，半径，颜色，线宽'''
cv2.circle(img,(260,260),100,(0,0,255),3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''绘制椭圆'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
'''参数为要绘制的图形，中心点坐标，长短轴长度，逆时针开始角度，顺时针开始角度，结束角度，颜色，线宽'''
cv2.ellipse(img,(260,260),(100,50),0,0,360,(0,0,255),-1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''绘制多边形'''
'''需要绘制各个顶点的坐标，并且数据类型必须是int32'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
'''需要绘制各个顶点的坐标，并且数据类型必须是int32'''
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,255)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''绘制多边形'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
'''需要绘制各个顶点的坐标，并且数据类型必须是int32'''
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,255)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''添加文字'''
import numpy as np
import cv2
img = np.zeros((512,512,3),np.uint8)
#设置字体
font = cv2.FONT_HERSHEY_SIMPLEX
#参数：要绘制的图像，文字内容，位置，字体，字体大小，颜色，线条
cv2.putText(img,'opencv',(0,200),font,4,(0,0,255),3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()












