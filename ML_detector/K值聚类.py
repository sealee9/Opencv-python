# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:28:26 2019

@author: Administrator
"""

'''
原理： 利用一个例子来说明
       一个公司要生产一批T恤，很明显要生产不同大小的T恤来满足不同客户的需求，这个公司收集了很多人的身高和体重
       信息，肯定不能把每个大小的T恤都生产出来，可以使用K值聚类的方法将所有人分为三组，这个算法可以找到一个最
       好的方法，并能覆盖所有人，如果不能覆盖所有人的话，那就只能把这些人分为更多组，可能是4或5组。
算法过程：
        是一个迭代的过程
        1、随机选取两个重心点，C1,C2，(有时可以选取数据中的两个点作为起始重心)
        2、计算每个点到这两个重心的距离，如果距离C1较近就标记为0，如果距离C2较近就标记为1
        3、重新计算标记为0的点的重心和标记为1的点的重心，并以这两个点更新重心点的位置
        4、重复步骤2，更新所有标记的点
        5、继续迭代步骤2和3，直到两个重心点的位置稳定下来(也可以设置迭代次数或者重心移动距离的阈值来终止迭代)
           此时这些点到它们相应重心的距离之和最小。
Opencv中K值聚类：
                cv2.kmeans()
                参数设置
                1.samples:应该是np.float32类型，每个特征应该放在一列
                2.nclusters:聚类的最终数目
                3.criteria:终止迭代的条件，是一个含有3个成员的元组(type,max_iter,epsilon)
                    type终止的类型有以下选择：
                                            cv2.TERM_CRITERIA_EPS，只有当精度epsilon满足条件时
                                            cv2.TERM_CRITERIA_MAX_ITER，只有当迭代次数满足时
                                            cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，以上两个任意一个满足就可以
                    max_iter:表示最大迭代次数
                    epsilon:精确度阈值
                4.attempts:使用不同的起始标记来执行算法的次数，算法会返回紧密度最好的标记
                5.flags：用来设置如何选择起始重心，通常有两个选择:cv2.KMEANS_PP_CENTERS和cv2.KMEANS_RANDOM_CENTERS
                
                输出参数：
                        1.compactness:紧密度，返回每个点到相应重心的距离的平方和
                        2.labels：标志数组，每个成员被标记为0、1等
                        3.centers：由聚类的中心组成的数组
                
'''


#代码1：仅有一个特征的数据
import cv2
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)
plt.hist(z,256,[0,256]),plt.show()
#设置算法迭代终止的条件
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)
A = z[labels==0]
B = z[labels==1]

plt.hist(A,256,[0,256],color='r')
plt.hist(B,256,[0,256],color='b')
plt.hist(centers,32,[0,256],color='y')
plt.show()


############################################################################
########################两个特征的数据

import cv2
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(25,50,(25,2))
y = np.random.randint(60,85,(25,2))
z = np.vstack((x,y))
z = np.float32(z)

#设置算法迭代终止的条件
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)


A = z[labels.ravel()==0]
B = z[labels.ravel()==1]

plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c='r')
plt.scatter(centers[:,0],centers[:,1],s=80,c='y',marker='s')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()

###############################################################################
####################颜色量化
'''
颜色量化就是减少图片中颜色数目的一个过程，减少图片中的颜色是为了内存消耗，有些设备的资源
有限，只能显示很少的颜色，在这种情况下就需要颜色量化，可以使用K值聚类的方法来进行颜色量化
由于图片有3个特征R、G、B，需要把图片数据变形为Mx3（M是图片中像素点的数目）的向量，聚类完
成后，用聚类中心值替换与其同组的像素值，这样结果图片就只含有指定数目的颜色。
'''
import numpy as np
import cv2

img = cv2.imread('../Opencv/images/home.jpg')
z = img.reshape((-1,3))
z = np.float32(z)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

for i in [2,4,8]:
    compactness,labels,centers = cv2.kmeans(z,i,None,criteria,10,flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]

    res2 = res.reshape((img.shape))

    cv2.imshow('res'+str(i),res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

























