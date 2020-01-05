# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:37:13 2019

@author: Administrator
"""

'''
K近邻是监督学习分类器
检测K个最近邻居，谁在这K个邻居中占据多数，那新的成员就属于那一类；
但是如果k等于4，两者出现均分的情况时无法准确分类
因此对距离进行赋予权重，近的点权重大，这被称为修改过的KNN

'''
##############################################################################
###############使用随机生成的数据进行KNN分类#####################################
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt


train_data = np.random.randint(0,100,(25,2)).astype(np.float32)
labels = np.random.randint(0,2,(25,1)).astype(np.float32)

red = train_data[labels.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

blue = train_data[labels.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')


newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(train_data,cv2.ml.ROW_SAMPLE,labels)
ret,result,neighbours,dist = knn.findNearest(newcomer,3)
print('result',result)
print('neighbours',neighbours)
print('distance',dist)
plt.show()

'''



#######################################################################################
#######################################################################################
##使用k近邻来进行手写数字的分类，文件中的digits.png图片上有5000个数字，我们要对图片中数字进行切分
##作为数据集，然后进行训练和分类

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

x = np.array(cells)  #(50,100,20,20)

train = x[:,:50].reshape(-1,400).astype(np.float32) #(2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32)#(2500,400)

k = np.arange(10)

train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighgours,dist = knn.findNearest(test,k=5)

matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)


#为了避免每次运行程序时，都要准备和训练分类器，我们最好把它保留，这样在下次运行时，只需要
#从文件中读取这些数据开始进行分类就可以了，numpy函数np.savetxt,np.load等可以帮助搞定这些
np.savez('knn_data.npz',train=train,train_labels=train_labels)
#下载数据
with np.load('knn_data.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']


###############################################################################
###########################英文字母的分类
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
#下载数据，把字母转换成数字
data = np.loadtxt('letter-recognition.data',dtype='float32',delimiter=',',
                  converters={0:lambda ch:ord(ch)-ord('A')})
train,test = np.vsplit(data,2)

responses,traindata = np.hsplit(train,[1])
labels,testdata = np.hsplit(test,[1])

knn = cv2.ml.KNearest_create()
knn.train(traindata,cv2.ml.ROW_SAMPLE,responses)
ret,result,neighbours,dist = knn.findNearest(testdata,k=5)

correct = np.count_nonzero(result==labels)
accuracy = correct*100.0/result.size
print(accuracy)
'''












