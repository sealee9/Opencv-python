# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:59:11 2019

@author: Administrator
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
import math
import time

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    """生成网络中经常用到的函数的默认参数"""
    batch_norm_params = {
        'decay': 0.9997,  #衰减系数
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }

    # slim.arg_scope可以给函数的参数自动赋予某些默认值，例如with slim.arg_scope([slim.conv2d,
    # slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay))这句会对
    # slim.conv2d, slim.fully_connected这两个函数的参数自动赋值，将参数weights_regularizer默认
    # 设置为slim.l2_regularizer(weight_decay)。此后不需要每次都设置参数了只需要在修改的时候设置
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 再对卷积层函数进行默认参数配置
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as sc:
            return sc

'''生成Inception V3网络的卷积部分'''
def inception_v3_base(inputs, scope=None):  #inputs表示输入的图片数据的张量，scope为包含了函数默认参数的环境
 
    end_points = {} #字典表，用来保存某些关键节点供之后使用
 
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
 
        '''定义前几层的卷积池化层'''
        #使用slim.arg_scope对slim.conv2d, slim.max_pool2d, slim.avg_pool2d的参数设置默认值
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],    #卷积，最大池化，平均池化
                            stride=1, padding='VALID'): #步长默认设为1，padding默认为VALID
 
            #定义卷积层：slim.conv2d(inputs, 输出的通道数, 卷积核尺寸, 步长, padding模式)
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1' )
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
 
        '''定义三个Inception模块组'''
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            '''定义第一个Inception模块组，包含三个结构类似的Inception Module'''
            #第一个Inception模块组的第一个Inception Module,有4个分支，从Branch_0到Branch_3
            with tf.variable_scope('Mixed_5b'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                #第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                #将四个分支的输出合并，由于步长皆为1且padding为SAME模式，所以图片尺寸没有缩小，只是通道数增加了，
                # 因此在第三个维度上合并，即输出通道上合并，64+64+96+32=256，所以最终尺寸为35*35*256
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            #第一个Inception模块组的第二个Inception Module,有4个分支，从Branch_0到Branch_3
            with tf.variable_scope('Mixed_5c'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                #第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #将四个分支的输出合并，64+64+96+64=288,所以最终尺寸35*35*288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            # 第一个Inception模块组的第三个Inception Module,有4个分支，从Branch_0到Branch_3
            #同第二个Inception Module
            with tf.variable_scope('Mixed_5d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支的输出合并，64+64+96+64=288,所以最终尺寸35*35*288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            '''定义第二个Inception模块组，共包含5个Inception Module'''
            #第二个Inception模块组的第一个Inception Module，有三个分支
            with tf.variable_scope('Mixed_6a'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                #将三个分支合并，每个分支中都有步长为2的，因此图片尺寸被压缩为一半即17*17，又384+96+288=768，所以尺寸为17*17*768
                net = tf.concat([branch_0, branch_1, branch_2], 3)
 
            # 第二个Inception模块组的第二个Inception Module，有四个分支
            with tf.variable_scope('Mixed_6b'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                #第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                #将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            # 第二个Inception模块组的第三个Inception Module，有四个分支
            with tf.variable_scope('Mixed_6c'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                #第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                #将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            # 第二个Inception模块组的第四个Inception Module，有四个分支
            #同第三个Inception Module
            with tf.variable_scope('Mixed_6d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            # 第二个Inception模块组的第五个Inception Module，有四个分支
            # 同第三个Inception Module
            with tf.variable_scope('Mixed_6e'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            #将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
            end_points['Mixed_6e'] = net
 
 
            '''定义第三个Inception模块组，共包含3个Inception Module'''
 
            #第三个Inception模块组的第一个Inception Module，有三个分支
            with tf.variable_scope('Mixed_7a'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                #将三个分支合并，步长为2，图片尺寸变为原来的一半，所以tensor的尺寸为8*8*(320+192+768)=8*8*1280
                net = tf.concat([branch_0, branch_1, branch_2], 3)
 
            # 第三个Inception模块组的第二个Inception Module，有四个分支
            with tf.variable_scope('Mixed_7b'):
                #第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                #第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  #8*8*(384+384)=8*8*768
                #第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  #8*8*(384+384)=8*8*768
                #第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                #将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            # 第三个Inception模块组的第三个Inception Module，有四个分支
            #同第二个Inception Module
            with tf.variable_scope('Mixed_7c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
 
            #返回这个Inception Module的结果作为该函数的结果
            return net, end_points
    
def inception_v3(inputs,
                 num_classes=1000, #最后分类的数量
                 is_training=True, #标志是否是训练过程，对Batch Normalization和Drop out有影响，只有在训练时这人两个才会被启用
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax, #分类函数
                 spatial_squeeze=True,  #出去维数是1的维度，例如5x3x1变为5x3
                 reuse=None,        #标志是否对网络和Variable进行重复使用
                 scope='InceptionV3'): #scope为包含了函数默认参数的环境
    with tf.variable_scope(scope, 'InceptionV3',
                           [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)

        # 辅助分类节点Auxiliaary Logits
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                             padding='VALID', scope='AvgPool_1a_5x5')
                aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                aux_logits = slim.conv2d(aux_logits, 768, [5, 5],
                                         weights_initializer=trunc_normal(0.01),
                                         padding='VALID', scope='Conv2d_2a_5x5')
                aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                                         activation_fn=None, normalizer_fn=None,
                                         weights_initializer=trunc_normal(0.001),
                                         scope='Conv2d_2b_1x1')
                if spatial_squeeze:
                    #消除1x1x1000中前两个维数1变为1000
                    aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatiaSqueeze')

                end_points['AuxLogits'] = aux_logits

        # 正常分类预测的逻辑
        with tf.variable_scope('Logits'):
            net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
            end_points['PreLogits'] = net
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            if spatial_squeeze:
                #消除1x1x1000中前两个维数1变为1000
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predications')

    return logits, end_points

def time_tensorflow_run(session, target, info_string):
    """
    评估Inception_V3每轮计算时间的函数
    :param session: TensorFlow的Session
    :param target: 需要预测的算子
    :param info_string: 测试的名称
    :return: 
    """
    num_steps_burn_in = 10  # 预热轮数，因为头几轮迭代有显存加载,cache命中等问题，所以不考虑
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 总时间平方和
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:  # 预热轮数之后再显示每轮消耗时间
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration  # 累加用于计算后面每轮耗时的均值
            total_duration_squared += duration * duration  # 累加用于计算后面每轮函数的标准差
    mn = total_duration / num_batches  # 计算每轮平均耗时
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)  # 计算标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(inputs, is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, logits, "Forward")
