# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:54:45 2019

@author: Administrator
"""

from datetime import datetime
import math
import time 
import tensorflow as tf

batch_size = 32
num_batches = 100

#定义一个用来显示网络每一层结构的函数，展示每一个卷积层或池化层输出的tensor的尺寸
#这个函数接受一个tensor作为输入
def print_activations(t):
    print(t.op.name,'',t.get_shape().as_list())

#设计AlexNet网络结构
def inference(images):
    parameters = []
    
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv1)
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
    print_activations(pool1)
    
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv2)
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)
    
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv3)
    
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv4)
    
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv5)
    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    print_activations(pool5)
    #return pool5,parameters

    #构造全连接层
    with tf.name_scope('f_c1') as scope:
        final_conv_shape = pool5.get_shape().as_list() 
        final_shape = final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]
        final_conv_out = tf.reshape(pool5,[final_conv_shape[0],final_shape])
        f_c1_weights = tf.Variable(tf.truncated_normal([final_shape,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        f_c1_biases = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        f_c1_out = tf.nn.relu(tf.add(tf.matmul(final_conv_out,f_c1_weights),f_c1_biases))
    print_activations(f_c1_out)
    
    with tf.name_scope('f_c2') as scope:
        f_c2_weights = tf.Variable(tf.truncated_normal([4096,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        f_c2_biases = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        f_c2_out = tf.nn.relu(tf.add(tf.matmul(f_c1_out,f_c2_weights),f_c2_biases))
    print_activations(f_c2_out)
    
    with tf.name_scope('f_c3') as scope:
        f_c3_weights = tf.Variable(tf.truncated_normal([4096,1000],dtype=tf.float32,stddev=1e-1),name='weights')
        f_c3_biases = tf.Variable(tf.constant(0.0,shape=[1000],dtype=tf.float32),trainable=True,name='biases')
        f_c3_out = tf.nn.relu(tf.add(tf.matmul(f_c2_out,f_c3_weights),f_c3_biases))
    print_activations(f_c3_out)
    
    return f_c3_out,parameters

#实现一个评估网络每轮计算时间的函数
def time_tensorflow_run(session,target,info_string):
    #定义预热轮数，作用是给程序热身    
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s:step %d,duration = %.3f'%(datetime.now(),i - num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration*duration
    #循环结束后，计算每轮迭代的平均时耗mn和标准差sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s:%s across %d steps,%.3f +/- %.3f sec / batch'%(datetime.now(),info_string,num_batches,mn,sd))
    
            
#主函数构建，首先使用with tf.Graph().as_default()定义默认的Graph方便后面使用
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
        f_c3_out,parameters = inference(images)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        time_tensorflow_run(sess,f_c3_out,"Forward")
        
        objective = tf.nn.l2_loss(f_c3_out)
        grad = tf.gradients(objective,parameters)
        
        time_tensorflow_run(sess,grad,"Forward-backward")

run_benchmark()
        
        
        
        
            
            
        


    
    
    
    
        
        
 