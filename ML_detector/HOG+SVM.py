# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:11:44 2019

@author: Administrator
"""

import cv2
import numpy as np
import random
 
#加载样本
def loadImageList(dirName,fileListPath):
    imageList = []
    file = open(dirName+r'/'+fileListPath)
    imageName = file.readline()
    while imageName != '':
        imageName = dirName+r'/'+imageName.split('/',1)[1].strip('\n')
        temp = cv2.imread(imageName)
        temp = cv2.resize(temp,(96,160))
        imageList.append(temp)
        imageName = file.readline()
    return imageList
#获取正样本，从（16， 16）截取大小为（128,64）的区域
def getPosSample(imageList):
	posList = []
	for i in range(len(imageList)):
		roi = imageList[i][16:16+128, 16:16+64]
		posList.append(roi);
	return posList
 
#获取负样本，从没有行人的图片中，随机裁剪出10张大小为（128, 64）的区域
def getNegSample(imageList):
	negList = []
	random.seed(1)
	for i in range(len(imageList)):
		for j in range(10):
			y = int(random.random() * (len(imageList[i]) - 128))
			x = int(random.random() * (len(imageList[i][0]) - 64))
			negList.append(imageList[i][y:y + 128, x:x + 64])
	return negList
	
#计算HOG特征
def getHOGList(imageList):
	HOGList = []
	hog = cv2.HOGDescriptor()
	for i in range(len(imageList)):
		gray = cv2.cvtColor(imageList[i], cv2.COLOR_BGR2GRAY)
		HOGList.append(hog.compute(gray))
	return HOGList
 
#获取检测子
def getHOGDetector(svm):
	sv = svm.getSupportVectors()
	rho, _, _ = svm.getDecisionFunction(0)
	sv = np.transpose(sv)
	return np.append(sv, [[-rho]], 0)
 
#获取Hard example
#def getHardExamples(negImageList, svm):
#	hardNegList = []
#	hog = cv2.HOGDescriptor()
#	hog.setSVMDetector(getHOGDetector(svm))
#	for i in range(len(negImageList)):
#	    rects, wei = hog.detectMultiScale(negImageList[i], winStride=(4, 4),padding=(8, 8), scale=1.05)
#	    for (x,y,w,h) in rects:
#		hardExample = negImageList[i][y:y+h, x:x+w]
#		hardNegList.append(cv2.resize(hardExample,(64,128)))
#	return hardNegList
def getHardExamples(negImageList,svm):
    hardNegList = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(getHOGDetector(svm))
    for i in range(len(negImageList)):
        rects,wei = hog.detectMultiScale(negImageList[i],winStride=(4,4),padding=(8,8),scale=1.05)
        for (x,y,w,h) in rects:
            hardExample = negImageList[i][y:y+h,x:x+w]
            hardNegList.append(cv2.resize(hardExample,(64,128)))
    return hardNegList
    
    
    
#非极大值抑制
def fastNonMaxSuppression(boxes, sc, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = sc
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]
 
#主程序
labels = []
posImageList = []
posList = []
posImageList = []
posList = []
hosList = []
tem = []
hardNegList = []
#加载含行人的图片
posImageList = loadImageList(r"G:/Opencv/INRIAPerson/Train", "pos.lst") 
print ("posImageList:", len(posImageList))
print(posImageList[0].shape)
#剪裁图片，获取正样本
posList = getPosSample(posImageList)
print ("posList", len(posList))
#获取正样本的HOG
hosList = getHOGList(posList)
print ("hosList", len(hosList))
#添加所有正样本对应的label
[labels.append(+1) for _ in range(len(posList))]
 
#加载不含行人的图片
negImageList = loadImageList(r"G:/Opencv/INRIAPerson/96X160H96/Train", "neg.lst") 
print ("negImageList:", len(negImageList))
#随机裁剪获取负样本
negList = getNegSample(negImageList)
print ("negList", len(negList))
#获取负样本HOG，并添加到整体HOG特征list中
hosList.extend(getHOGList(negList))
print ("hosList", len(hosList))
#添加所有负样本对应的label
[labels.append(-1) for _ in range(len(negList))]
print ("labels", len(labels))
####################至此得到SVM的所有特征和label（不含hard example）######################
 
 
#创建svm分类器，参数设置 
#################################################################
#-d degree：核函数中的degree设置(针对多项式核函数)(默认3)
 
#-g r(gama)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/ k)
 
#-r coef0：核函数中的coef0设置(针对多项式/sigmoid核函数)((默认0)
 
#-c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)
 
#-n nu：设置v-SVC，一类SVM和v- SVR的参数(默认0.5)
 
#-p p：设置e -SVR 中损失函数p的值(默认0.1)
 
#-m cachesize：设置cache内存大小，以MB为单位(默认40)
 
#-e eps：设置允许的终止判据(默认0.001)
 
#-h shrinking：是否使用启发式，0或1(默认1)
 
#-wi weight：设置第几类的参数C为weight*C(C-SVC中的C)(默认1)
 
#-v n: n-fold交互检验模式，n为fold的个数，必须大于等于2
##################################################################################
 
svm = cv2.ml.SVM_create()
svm.setCoef0(0.0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-3)#终止条件
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
svm.setC(0.01)  # From paper, soft classifier 软间隔
svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))
 
#根据初始训练结果获取hard example
hardNegList = getHardExamples(negImageList, svm)
hosList.extend(getHOGList(hardNegList))
print ("hosList=====", len(hosList))
[labels.append(-1) for _ in range(len(hardNegList))]
####################至此得到SVM的所有特征和label（含hard example）######################
####################实测添加hard example可以很大提高检测的查准率#########################
 
#添加hard example后，重新训练
svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))
 
#保存模型
hog = cv2.HOGDescriptor()
hog.setSVMDetector(getHOGDetector(svm))
hog.save('myHogDector.bin')
 
#行人检测
#hog.load('myHogDector.bin') #因为在同一个文件中，不需要加载模型
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')
image = cv2.imread("images/2222.png")
image = cv2.resize(image,(96,160))
#cv2.imshow("image", image)
#cv2.waitKey(0)
rects, scores = hog.detectMultiScale(image, winStride=(4,4),padding=(8, 8), scale=1.05)

#fastNonMaxSuppression第一个参数
for i in range(len(rects)):
    r = rects[i]
    rects[i][2] = r[0] + r[2]
    rects[i][3] = r[1] + r[3]
 
#fastNonMaxSuppression第二个参数
sc = [score[0] for score in scores]
sc = np.array(sc)
 
pick = []
print('rects_len',len(rects))
pick = fastNonMaxSuppression(rects, sc, overlapThresh = 0.3)
print('pick_len = ',len(pick))
 
 
for (x, y, xx, yy) in pick:
	print (x, y, xx, yy)
	cv2.rectangle(image, (int(x), int(y)), (int(xx), int(yy)), (0, 0, 255), 2)
cv2.namedWindow('a',cv2.WINDOW_NORMAL)
cv2.imshow('a', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

