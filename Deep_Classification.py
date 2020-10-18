# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:46:42 2019

@author: jason
"""

import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
Start = time.time() 
###############################################################################
# 資料集載入
EE_data = pd.read_csv('ionosphere_csv.csv')
(rows,columns)=EE_data.shape
rows_number = list(range(rows))
###############################################################################
learning_rate = 1/10**5
Training_epochs = 100
Training_time = 500
input_number=34
#neurons_number
neurons_1_number=10
neurons_2_number=20
neurons_3_number=10
neurons_4_number=2
train=rows*80//100
test=rows*20//100+1
train_loss_col = []
test_loss_col = []
train_layer_4 = []
train_y = []
test_y = []
test_layer_4 = []
train_g=[]
train_b=[]
test_g=[]
test_b=[]
T_t=Training_time-1
###########################################################################################
###########################################################################################
w1=np.random.randint(1,10,[input_number,neurons_1_number])/1000 #(1,10）以內的X行X列隨機整數
w2=np.random.randint(1,10,[neurons_1_number,neurons_2_number])/1000
w3=np.random.randint(1,10,[neurons_2_number,neurons_3_number])/1000
w4=np.random.randint(1,10,[neurons_3_number,neurons_4_number])/1000
###########################################################################################
b1=np.random.randint(1,10,[neurons_1_number,1])/100 #(1,10）以內的X行X列隨機整數
b2=np.random.randint(1,10,[neurons_2_number,1])/100
b3=np.random.randint(1,10,[neurons_3_number,1])/100
b4=np.random.randint(1,10,[neurons_4_number,1])/100
###########################################################################################
data = np.eye(rows,columns+1)*0
Train_data=np.eye(train,columns+1)*0
Test_data=np.eye(test,columns+1)*0
###########################################################################################
# 資料集整理
for r_index in rows_number:
    data[r_index,0]=EE_data['a01'][r_index]
    data[r_index,1]=EE_data['a02'][r_index]
    data[r_index,2]=EE_data['a03'][r_index]
    data[r_index,3]=EE_data['a04'][r_index]
    data[r_index,4]=EE_data['a05'][r_index]
    data[r_index,5]=EE_data['a06'][r_index]
    data[r_index,6]=EE_data['a07'][r_index]
    data[r_index,7]=EE_data['a08'][r_index]
    data[r_index,8]=EE_data['a09'][r_index]
    data[r_index,9]=EE_data['a10'][r_index]
    data[r_index,10]=EE_data['a11'][r_index]
    data[r_index,11]=EE_data['a12'][r_index]
    data[r_index,12]=EE_data['a13'][r_index]
    data[r_index,13]=EE_data['a14'][r_index]
    data[r_index,14]=EE_data['a15'][r_index]
    data[r_index,15]=EE_data['a16'][r_index]
    data[r_index,16]=EE_data['a17'][r_index]
    data[r_index,17]=EE_data['a18'][r_index]
    data[r_index,18]=EE_data['a19'][r_index]
    data[r_index,19]=EE_data['a20'][r_index]
    data[r_index,20]=EE_data['a21'][r_index]
    data[r_index,21]=EE_data['a22'][r_index]
    data[r_index,22]=EE_data['a23'][r_index]
    data[r_index,23]=EE_data['a24'][r_index]
    data[r_index,24]=EE_data['a25'][r_index]
    data[r_index,25]=EE_data['a26'][r_index]
    data[r_index,26]=EE_data['a27'][r_index]
    data[r_index,27]=EE_data['a28'][r_index]
    data[r_index,28]=EE_data['a29'][r_index]
    data[r_index,29]=EE_data['a30'][r_index]
    data[r_index,30]=EE_data['a31'][r_index]
    data[r_index,31]=EE_data['a32'][r_index]
    data[r_index,32]=EE_data['a33'][r_index]
    data[r_index,33]=EE_data['a34'][r_index]
    if EE_data['class'][r_index]=='g' :   
        data[r_index,34]=1
        data[r_index,35]=0
    if EE_data['class'][r_index]=='b' :   
        data[r_index,34]=0
        data[r_index,35]=1   
random.shuffle(data)
Train_data[0:train,0:36]=data[0:train,0:36]
Test_data[0:test,0:36]=data[train:rows,0:36]
###########################################################################################
for t in range(Training_time):
###########################################################################################    
    for i in range(0,train):
#    for i in range(0, 50):
#        i=1
        inp=Train_data[i,0:34].T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=Train_data[i,34:36]
        if t >= T_t :
            if y[0] >= y[1]:
                train_g.append(layer_4)
            else:
                train_b.append(layer_4)  
        # Compute and print loss
        train_loss = np.square(layer_4 - y).sum() # loss function
        train_loss_col.append(train_loss)
        train_y.append(y)
        train_layer_4.append(layer_4)
    #    print(i, loss, layer_4)
    #################################################################################
        # Backprop to compute gradients of weights with respect to loss
        grad_layer_4 = 2.0 * (layer_4 - y) # the last layer's error
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
        # Update weights   
        w1 = w1-(learning_rate * w1*(grad_w1))
        w2 = w2-(learning_rate * w2*(grad_w2))
        w3 = w3-(learning_rate * w3*(grad_w3))
        w4 = w4-(learning_rate * w4*(grad_w4))
    #################################################################################
        # Backprop to compute gradients of bias with respect to loss
        grad_b4 = layer_3.T.dot(a)    
        grad_b3 = layer_2.T.dot(b)
        grad_b2 = layer_1.T.dot(c)   
        grad_b1 = inp.T.dot(d)
        # Update bias    
        b1 = b1-(learning_rate * b1*(sum(grad_b1)))
        b2 = b2-(learning_rate * b2*(sum(grad_b2)))
        b3 = b3-(learning_rate * b3*(sum(grad_b3)))
        b4 = b4-(learning_rate * b4*(sum(grad_b4)))
#    if train_loss <= 0.001 : break
#################################################################################
plt.plot(train_loss_col)
plt.show()
#################################################################################
#################################################################################
array=np.array(train_g)
(g,gg)=array.shape
trxg = np.eye(g,1)*0
tryg = np.eye(g,1)*0
trxg=array[0:g,0]
tryg=array[0:g,1]
(b,bb)=array.shape
array=np.array(train_b)
trxb = np.eye(b,1)*0
tryb = np.eye(b,1)*0
trxb=array[0:b,0]
tryb=array[0:b,1]
plt.figure ( figsize = ( 12 , 7 )) 
plt.plot(trxg*1000, tryg*1000, "ro", label = "$g$" )
plt.plot(trxb*1000, tryb*1000, "bo", label = "$b$")
plt.title ( "Training" ) 
plt.legend ()
plt.show()
###########################################################################################
for i in range(0,test):
#    for i in range(0, 50):
        inp=Test_data[i,0:34].T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=Test_data[i,34:36]
        if y[0] >= y[1]:
            test_g.append(layer_4)
        else:
            test_b.append(layer_4)          
        # Compute and print loss
        test_loss = np.square(layer_4 - y).sum() # loss function
        test_loss_col.append(test_loss)
        test_y.append(y)
        test_layer_4.append(layer_4)
    #    print(i, loss, layer_4)
###########################################################################################
array=np.array(test_g)
(g,gg)=array.shape
texg = np.eye(g,1)*0
teyg = np.eye(g,1)*0
texg=array[0:g,0]
teyg=array[0:g,1]
(b,bb)=array.shape
array=np.array(test_b)
texb = np.eye(b,1)*0
teyb = np.eye(b,1)*0
texb=array[0:b,0]
teyb=array[0:b,1]
plt.figure ( figsize = ( 12 , 7 )) 
plt.plot(texg*1000, teyg*1000, "ro", label = "$class 1 : g $" )
plt.plot(texb*1000, teyb*1000, "bo", label = "$class 2 : b $")
plt.title ( "Testing" ) 
plt.legend ()
plt.show()
###########################################################################################
End = time.time() 
print("Total %f sec" % (End - Start)) 