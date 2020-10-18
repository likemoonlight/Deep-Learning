# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:42:16 2019

@author: jason
"""
import time
import cv2
import glob as gb
import numpy as np
import pandas as pd
import random
from scipy.misc import imsave
import matplotlib.pyplot as plt
from PIL import Image 
import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data
#################################################################################################
Start = time.time() 
######################
## The MNIST data
#mnist = input_data.read_data_sets ( "MNIST_data/" , one_hot=True )
#training_data = mnist.train.images
#training_label = mnist.train.labels
#testing_data = mnist.test.images
#testing_label = mnist.test.labels
#z = training_data[1].reshape(28,28)
##########################################################################################################################################
#The CIFAR-10 dataset
def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict
#########################################################################################################################################
for i in range(1, 6):# 讀取當前目錄下的data_batch12345檔案，dataName其實也是data_batch檔案的路徑，本文和指令碼檔案在同一目錄下。
    batch_1=load_file('C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/data_batch_'+ str(i))
    for j in range(1, 6):
        dataName = "C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/data_batch_" + str(j)  
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")    
        for k in range(0, 10000):
            img = np.reshape(Xtr['data'][k], (3, 32, 32))  # Xtr['data']為圖片二進位制資料
            img = img.transpose(1, 2, 0)  # 讀取image
            picName = 'C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/train/' + str(Xtr['labels'][k]) + '_' + str(k + (j - 1)*10000) + '.jpg'  
            imsave(picName, img)# Xtr['labels']為圖片的標籤，值範圍0-9，本文中，train資料夾需要存在，並與指令碼檔案在同一目錄下。
        print(dataName + " loaded.")
################################################################################################
test_batch=load_file('C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/test_batch')
for i in range(0, 10000):
    img = np.reshape(test_batch['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/test/' + str(test_batch['labels'][i]) + '_' + str(i) + '.jpg'
    imsave(picName, img)
print("test_batch loaded.")
#################################################################################################
#Returns a list of all folders with participant numbers
Train_img_path = gb.glob("C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/train\\*.jpg") 
Test_img_path = gb.glob("C:/Users/jason/Desktop/cifar-10-python/cifar-10-batches-py/test\\*.jpg") 
label = np.identity(10) 
z  = cv2.imread(Train_img_path[1])/255
##################################################################################################
##########################################################################################################################################
#################################################################################################
pooling=2
filter_num1=5
filter_w1=3
filter_h1=3
##########################
filter_num2=6
filter_w2=3
filter_h2=3
##########################
learning_rate = 1/10**5
Training_epochs = 100
Training_time = 1000
train=10000
test=10000
#################################################################################################
filter1 = np.random.randint(-10,10,[filter_num1,filter_h1,filter_w1])/10
filter2 = np.random.randint(-10,10,[filter_num2,filter_h2,filter_w2])/10
##########################################################################################################################################
def feature_map(image_source,filter):
#    image_source = y
#    filter = filter1
    dim=image_source.shape #(height, width, channel)
    F = filter.shape
    f1,f2,f3 = F
    if len(dim) != 2:
        C, H, W = dim
        feature_img = np.zeros((f1*C,H+3-f3,W+3-f2),dtype=float)
        new_img = np.zeros((C,H+2,W+2),dtype=float)
        new_img[0:C,1:H+1,1:W+1] = image_source
        H_numberli = list(range(H+3-f3))
        W_numberli = list(range(W+3-f2))
        C_numberli = list(range(C))
        F_numberli = list(range(f1))
        f = 0 ; c = 0 ; g = 0
        for f_index in F_numberli:
            for c_index in C_numberli:
                if f != f_index or c != c_index :
                            g = g + 1
                            f=f_index;c=c_index
                for h_index in H_numberli:        
                    for w_index in W_numberli:
                        feature_img[g,h_index,w_index]= sum(sum(filter[f_index].dot(new_img[c_index,h_index:h_index+f3,w_index:w_index+f2]))) 
    else:
        H, W = dim
        feature_img = np.zeros((f1,H+3-f3,W+3-f2),dtype=float)
        new_img = np.zeros((H+2,W+2),dtype=float)
        new_img[1:H+1,1:W+1] = image_source
        H_numberli = list(range(H+3-f3))
        W_numberli = list(range(W+3-f2))
        F_numberli = list(range(f1))
        for h_index in H_numberli:        
            for w_index in W_numberli:
                for f_index in F_numberli:
                    feature_img[f_index,h_index,w_index]= sum(sum(filter[f_index].dot(new_img[h_index:h_index+f3,w_index:w_index+f2])))  
    return  feature_img
#################################################
#map=feature_map(y,filter1)
#im=Image.fromarray(map[0]*255) # numpy 转 image类
#im.show()
########################################################################################################################################
def reLu(reLu_map):
#    reLu_map=map
    R=reLu_map.shape
    r1,r2,r3 = R
    map_reLu = np.zeros((r1,r2,r3),dtype=float)
    map_reLu = np.maximum(reLu_map, 0)
    return map_reLu
#################################################
#re=reLu(map)
#im=Image.fromarray(re[0]*255) # numpy 转 image类
#im.show()
#######################################################################################################################################
def max_pooling(im,pooling):
#    im=step1_relu
    I=im.shape #(height, width, channel)
    i1,i2,i3 = I
    pooling_img = np.zeros((i1,i2//pooling,i3//pooling),dtype=float)    
    H_numberli = list(range(i2//pooling))
    W_numberli = list(range(i3//pooling))
    M_numberli = list(range(i1))
    for m_index in M_numberli:
        for h_index in H_numberli:        
            for w_index in W_numberli:            
                pooling_img[m_index,h_index,w_index]= np.max(im[m_index,
                           h_index*pooling:h_index*pooling+pooling,w_index*pooling:w_index*pooling+pooling])  
    return  pooling_img
#################################################
#pl=max_pooling(re,pooling)
#im=Image.fromarray(pl[0]*255) # numpy 转 image类
#im.show()
######################################################################################################################################
def Convolution():
    step1_fm = feature_map(z,filter1)
    step1_relu = reLu(step1_fm)
    step1_pool = max_pooling(step1_relu,pooling)
    step2_fm = feature_map(step1_pool,filter2)
    step2_relu = reLu(step2_fm)
    step2_pool = max_pooling(step2_relu,pooling)
    return step2_pool
###################################################
###################################################
final=Convolution()
#im=Image.fromarray(final[0]*255) # numpy 转 image类
#im.show()
f1,f2,f3=final.shape
#z = final.reshape(1,f1*f2*f3)
######################################################################################################################################
def compute_cost(AL, Y): 
#    Y=layer_4
#    AL=y
    m = Y.shape[0] # Compute loss from AL and y 
    logprobs =np.multiply(np.log(AL),Y) + np.multiply(1 - Y, np.log(1 - AL)) # cross-entropy cost 
    cost = - np.sum(logprobs) / m 
    cost = np.squeeze(cost) 
    return cost
######################################################################################################################################
######################################################################################################################################
#neurons_number
input_number=f1*f2*f3
neurons_1_number=300
neurons_2_number=500
neurons_3_number=200
neurons_4_number=10
###################################################
Test_total_rate = [];Train_total_rate = []
Train_total_loss = [];Test_total_loss = []
train_loss_col = [];test_loss_col = []
train_Cross_Entropy = [];test_Cross_Entropy = []
train_layer_4 = [];test_layer_4 = []
train_y = [];test_y = []
Train_rate = np.zeros((Training_time,train),dtype=float)
###########################################################################################
w1=np.random.randint(-10,10,[input_number,neurons_1_number])/1000 #(1,10）以內的X行X列隨機整數
w2=np.random.randint(-10,10,[neurons_1_number,neurons_2_number])/1000
w3=np.random.randint(-10,10,[neurons_2_number,neurons_3_number])/1000
w4=np.random.randint(-10,10,[neurons_3_number,neurons_4_number])/1000
###########################################################################################
b1=np.random.randint(-10,10,[neurons_1_number,1])/100 #(1,10）以內的X行X列隨機整數
b2=np.random.randint(-10,10,[neurons_2_number,1])/100
b3=np.random.randint(-10,10,[neurons_3_number,1])/100
b4=np.random.randint(-10,10,[neurons_4_number,1])/100
###########################################################################################
##Training
tt = 0 ; ii = 0 ; jj = 0 ; g = 1 ; k = 1 ;
grad_1 = np.zeros((train,neurons_1_number),dtype=float)
grad_2 = np.zeros((train,neurons_2_number),dtype=float)
grad_3 = np.zeros((train,neurons_3_number),dtype=float)
grad_4 = np.zeros((train,neurons_4_number),dtype=float)
###########################################################################################
for t in range(Training_time):
#############################################   
    for i in range(0,train):
#############################################
#        # MNIST data
#        z = training_data[i].reshape(28,28)
#        final=Convolution()
       #################################### 
        # CIFAR-10 data
        a = Train_img_path[i]
        z  = cv2.imread(a)/255
        final=Convolution()
#############################################
#        r_1=(np.dot(inp.T,w1)+b1.T)        
        inp=final.reshape(1,input_number).T
        # Forward pass: compute predicted y
        r_1=(np.dot(inp.T,w1)+b1.T)
        layer_1=np.maximum(r_1.T, 0)
        r_2=(np.dot(layer_1.T,w2)+b2.T)
        layer_2=np.maximum(r_2.T, 0)
        r_3=(np.dot(layer_2.T,w3)+b3.T)
        layer_3=np.maximum(r_3.T, 0)
        r_4=(np.dot(layer_3.T,w4)+b4.T)
        layer_4=np.maximum(r_4.T, 0)
        final_out = tf.nn.softmax(layer_4.T)
#        y=training_label[i]# For MNIST data
        y=label[int(a[65])]# For CIFAR-10 data
        Cross_Entropy = compute_cost(final_out,y)
        aa=layer_4.max();bb=layer_4/aa
        cc = bb.copy();cc[cc < 1] = 0
        loss = y-cc.T ; Loss = loss*loss ; total_loss = Loss.sum()
        # Compute and print loss
        train_loss = np.square(final_out-y).sum() # loss function
        train_loss_col.append(train_loss)
        train_Cross_Entropy.append(Cross_Entropy)
        Train_total_loss.append(total_loss) 
        train_y.append(y)
        train_layer_4.append(layer_4)
      ###################################################################
        if tt != t or ii != i :
            g = g + 1 ; tt=t; ii=i
        array=np.array(Train_total_loss)
        total_loss = array.sum()/2
        Train_total_loss_rate = total_loss/g
        Train_total_rate.append(1-Train_total_loss_rate)
        Train_rate[t,i]=(1-Train_total_loss_rate)      
      ###################################################################
        # Backprop to compute gradients of weights with respect to loss
        grad_layer_4 = 2 * ( layer_4.T-y) # the last layer's error    
#        grad_layer_4 = Cross_Entropy * (  layer_4.T-y) # the last layer's error   
        a=np.ones((neurons_3_number,1))*grad_layer_4
        grad_w4 = layer_3.T.dot(a)    
        grad_layer_3 = grad_layer_4.dot(w4.T) # the second laye's error 
        grad_h3 = grad_layer_3.copy()   
        grad_h3[layer_3 < 0] = 0  # the derivate of ReLU
        b=np.ones((neurons_2_number,1))*grad_h3
        grad_w3 = layer_2.T.dot(b)
        grad_layer_2 = grad_layer_3.dot(w3.T) # the second laye's error 
        grad_h2 = grad_layer_2.copy()
        grad_h2[layer_2 < 0] = 0  # the derivate of ReLU
        c=np.ones((neurons_1_number,1))*grad_h2
        grad_w2 = layer_1.T.dot(c)   
        grad_layer_1 = grad_layer_2.dot(w2.T) # the second laye's error 
        grad_h1 = grad_layer_1.copy()
        grad_h1[layer_1 < 0] = 0  # the derivate of ReLU
        d=np.ones((input_number,1))*grad_h1
        grad_w1 = inp.T.dot(d)      
        grad_1[i,0:neurons_1_number] = grad_w1
        grad_2[i,0:neurons_2_number] = grad_w2
        grad_3[i,0:neurons_3_number] = grad_w3
        grad_4[i,0:neurons_4_number] = grad_w4
    grad__1=sum(grad_1)/train
    grad__2=sum(grad_2)/train
    grad__3=sum(grad_3)/train
    grad__4=sum(grad_4)/train
    # Update weights   
    w1 = w1-(learning_rate * w1*(grad__1))
    w2 = w2-(learning_rate * w2*(grad_w2))
    w3 = w3-(learning_rate * w3*(grad_w3))
    w4 = w4-(learning_rate * w4*(grad_w4))
 ##################################################################
    # Backprop to compute gradients of bias with respect to loss
    grad_b4 = layer_3.T.dot(a)    
    grad_b3 = layer_2.T.dot(b)
    grad_b2 = layer_1.T.dot(c)   
    grad_b1 = inp.T.dot(d)
    # Update bias    
    b1 = b1-(learning_rate * b1*((grad_b1.T)))
    b2 = b2-(learning_rate * b2*((grad_b2.T)))
    b3 = b3-(learning_rate * b3*((grad_b3.T)))
    b4 = b4-(learning_rate * b4*((grad_b4.T)))
########################################################################
###########################################################################################
#Testing
###########################################################################################
    for j in range(0,test):
    #    for i in range(0, 50):
    ############################################### 
#        # MNIST data
#        z = testing_data[i].reshape(28,28)
#        final=Convolution()
       #################################### 
#         CIFAR-10 data
        a = Test_img_path[j]
        z  = cv2.imread(a)/255
        final=Convolution()
###############################################
        inp=final.reshape(1,input_number).T    
        # Forward pass: compute predicted y
        r_1=(np.dot(inp.T,w1)+b1.T)
        layer_1=np.maximum(r_1.T, 0)
        r_2=(np.dot(layer_1.T,w2)+b2.T)
        layer_2=np.maximum(r_2.T, 0)
        r_3=(np.dot(layer_2.T,w3)+b3.T)
        layer_3=np.maximum(r_3.T, 0)
        r_4=(np.dot(layer_3.T,w4)+b4.T)
        layer_4=np.maximum(r_4.T, 0)
        final_out = tf.nn.softmax(layer_4.T)
#        y=testing_label[j]# For MNIST data
        y=label[int(a[64])]# For CIFAR-10 data
        Cross_Entropy = compute_cost(final_out,y)
        aa=layer_4.max();bb=layer_4/aa
        cc = bb.copy();cc[cc < 1] = 0
        loss = y-cc.T ; Loss = loss*loss ; total_loss = Loss.sum()
        # Compute and print loss
        test_loss = np.square(layer_4.T - y).sum() # loss function
        test_loss_col.append(test_loss)
        test_Cross_Entropy.append(Cross_Entropy)
        Test_total_loss.append(total_loss)   
        test_y.append(y)
        test_layer_4.append(layer_4)
   ###############################################   
        if tt != t or jj != j :
            k = k + 1 ; tt=t; jj=j
        array=np.array(Test_total_loss)
        total_loss = array.sum()/2
        Test_total_loss_rate = total_loss/(k)
        Test_total_rate.append(1-Test_total_loss_rate)
###############################################################################          
Rate=sum(Train_rate)/Training_time
plt.plot(Rate)
plt.show()
###########################################################################################
###############################################################################
###############################################################################
plt.figure ( figsize = ( 15 , 7 )) 
plt.plot ( Train_total_rate , label = "$Training Accuracy$" , color = "red" , linewidth = 1 )
#plt.plot ( Train3 , label = "$Training Accuracy$" , color = "red" , linewidth = 1 ) 
plt.plot ( Test_total_rate , "b" , label = "$Testing Accuracy$" , linewidth = 1) 
#plt.plot (  TT , "b" , label = "$Testing Accuracy$" , linewidth = 1) 
#plt.plot ( train_Cross_Entropy , "g" , label = "$Train Cross Entropy$" , linewidth = 1)
plt.xlabel ( "Time(s)" ) 
plt.ylabel( "Accuracy Rate" ) 
plt.title ( "Training Accuracy" ) 
plt.legend () 
plt.show ()
###############################################################################
plt.figure ( figsize = ( 15 , 7 )) 
plt.plot ( train_Cross_Entropy , "g" , label = "$Cross Entropy$" , linewidth = 1) 
#plt.plot ( test_y , "b" , label = "$label$" , linewidth = 1) 
plt.xlabel ( "Time(s)" ) 
plt.ylabel( "Loss" ) 
plt.title ( "Learning Curve" ) 
plt.legend () 
plt.show ()
###########################################################################################
step1_fm = feature_map(z,filter1)
step1_relu = reLu(step1_fm)
step1_pool = max_pooling(step1_relu,pooling)
step2_fm = feature_map(step1_pool,filter2)
c1,c2,c3=step1_fm.shape
con_1=step1_fm.reshape(1,c1*c2*c3).T
c1,c2,c3=step2_fm.shape
con_2=step2_fm.reshape(1,c1*c2*c3).T
#################################################
z  = cv2.imread(a)# For CIFAR-10 data
im=Image.fromarray(z*255) # numpy 转 image类
im.show()
im=Image.fromarray(step1_pool[3]*255) 
im.show()
im=Image.fromarray(step2_fm[7]*255) 
im.show()
###############################################  
bins = np.arange( min(con_1) , max(con_1) , (max(con_1)-min(con_1))*0.01 )
plt.hist(con_1, bins) 
plt.xlabel ( "Value" ) 
plt.ylabel( "Number" ) 
plt.title("Histogram of Conv 1") 
plt.show()
###############################################  
bins = np.arange( min(con_2) , max(con_2) , (max(con_2)-min(con_2))*0.01 )
plt.hist(con_2, bins) 
plt.xlabel ( "Value" ) 
plt.ylabel( "Number" ) 
plt.title("Histogram of Conv 2") 
plt.show()
###############################################  
c1,c2=r_1.shape
r1=r_1.reshape(1,c1*c2).T
bins = np.arange( min(r1) , max(r1) , (max(r1)-min(r1))*0.01 )
plt.hist(r1, bins) 
plt.xlabel ( "Value" ) 
plt.ylabel( "Number" ) 
plt.title("Histogram of densel") 
plt.show()
###############################################  
bins = np.arange( min(layer_4) , max(layer_4) , (max(layer_4)-min(layer_4))*0.1 )
plt.hist(layer_4, bins) 
plt.xlabel ( "Value" ) 
plt.ylabel( "Number" ) 
plt.title("Histogram of output") 
plt.show()
###########################################################################################
End = time.time() 
print("Total %f sec" % (End - Start)) 

