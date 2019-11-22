# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:59:28 2019

@author: Administrator
"""

'''
GrabCut算法原理，使用该算法提取图像的前景
创建一个交互式程序完成前景提取

原理：   1.使用一个高斯混合模型(GMM)对前景和背景建模
        2.根据输入，GMM会学习并创建新的像素分布，对那些分类未知的区域，可以根据他们与已知分类的像素
          关系来进行分类（像是在做聚类操作）
        3.上面的操作会根据像素分布创建一个图，图中的节点就是像素，除了像素点做节点外还有两个节点
          Source_node和Sink_node,所有的前景像素都和Source_node相连，所有的背景像素都和Sink_node相连
        4.将像素连接到Sink_node，Source_node的权重由它们属于前景或者背景的概率决定，两个像素之间
           的权重由边的信息或者两个像素的相似性来决定，如果两个像素的颜色有很大的不同，则它们之间的边的权重会很小
        5.使用mincut算法对上面得到的图进行分割，它会根据最低成本方程将图分为Source_node和Sink_node,
           成本方程就是被剪掉的所有边的权重之和，在裁剪之后，所有连接到Source_node的像素被认为是前景
           所有连接到Sink_node的像素被认为是背景
        6.继续上面的过程直到分类收敛


Opencv中提供了cv2.grabCut()函数
参数： 1.输入图像
      2.mask:掩模图像，用来确定哪些是背景区域和前景区域，可能是前景/背景，可以设置为cv2.GC_BGD,cv2.GC_FGD
         cv2.GC_PR_BGD,cv2.GC_PR_FGD，也可以直接输入成0，1，2，3
      3.rect 包含前景的矩形，格式是(x,y,w,h)
      4.bgdModel,fgdModel 算法内部使用的数组，只需要创建两个大小为（1，65），数据类型np.float
      5.iterCount 算法的迭代次数
      6.mode可以设置为cv2.GC_INIT_WITH_RECT或cv2.GC_INIT_WITH_MASK,也可以联合使用，这是用来
         确定我们进行修改的方式，矩形模式或者掩模模式
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
cv2.imwrite('messi_.jpg',mask)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]   #np.newaixs的作用是：给数组增加维度，比如原数组是（5，），加上这个后是（5，1），方便后面的权重矩阵相乘

plt.imshow(img),plt.colorbar(),plt.show




#上面的结果并不理想，头发部分没有显示出来，还有一些草地部分显示出来了
#手动在绘图软件中用白色线标换出头发部位和球的部位
#其他部分用灰色填充
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
newmask = cv2.imread('messi.jpg',0)
# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 1
mask[newmask == 255] = 0
mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()














