# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:29:56 2019

@author: Administrator
"""

'''
import cv2
import numpy as np
#首先设置一个鼠标回调函数
def draw_circle(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(0,0,255),-1)
#创建图像与窗口并将窗口与回调函数绑定
img = np.zeros((255,255,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey()==27:
        break
cv2.destroyAllWindows()
'''

'''按下左键并移动绘制矩形，按下左键移动并按下m时绘制圆形'''
import numpy as np
import cv2

drawing = True
mode = True
ix,iy = -1,-1

'''设置鼠标回调函数'''
def draw_circle(event,x,y,flags,param):
    global drawing,mode,ix,iy
    if event==cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        drawing = True
    #如果鼠标左键按下并且移动时
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),20,(0,0,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)    
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)
    if k==ord('m'):
        mode = not mode
    elif k==27:
        break
cv2.destroyAllWindows()   
        












