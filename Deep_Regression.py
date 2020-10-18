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
EE_data = pd.read_csv('EnergyEfficiency_data.csv')
(rows,columns)=EE_data.shape
rows_number = list(range(rows))
###############################################################################
learning_rate = 1/10**10
Training_epochs = 100
Training_time = 1000
input_number=16
#neurons_number
neurons_1_number=64
neurons_2_number=128
neurons_3_number=32
neurons_4_number=1
train=rows*75//100
test=rows*25//100
train_loss_col = []
test_loss_col = []
train_layer_4 = []
train_y = []
test_y = []
test_layer_4 = []
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
data = np.eye(rows,columns+7)*0
Train_data=np.eye(rows*75//100,columns+7)*0
Test_data=np.eye(rows*25//100,columns+7)*0
###########################################################################################
# 資料集整理
for r_index in rows_number:
    data[r_index,0]=EE_data['Relative Compactness'][r_index]
    data[r_index,1]=EE_data['Surface Area'][r_index]#/80.85
    data[r_index,2]=EE_data['Wall Area'][r_index]#/41.65
    data[r_index,3]=EE_data['Roof Area'][r_index]#/22.05
    data[r_index,4]=EE_data['Overall Height'][r_index]#/0.7  
    if EE_data['Orientation'][r_index]==2 :   
        data[r_index,5]=1
        data[r_index,6]=0
        data[r_index,7]=0
        data[r_index,8]=0
    if EE_data['Orientation'][r_index]==3 :   
        data[r_index,5]=0
        data[r_index,6]=1
        data[r_index,7]=0
        data[r_index,8]=0    
    if EE_data['Orientation'][r_index]==4 :  
        data[r_index,5]=0
        data[r_index,6]=0
        data[r_index,7]=1
        data[r_index,8]=0   
    if EE_data['Orientation'][r_index]==5 :   
        data[r_index,5]=0
        data[r_index,6]=0
        data[r_index,7]=0
        data[r_index,8]=1    
    data[r_index,9]=EE_data['Glazing Area'][r_index]
    if EE_data['Glazing Area Distribution'][r_index]==0 :   
        data[r_index,10]=0
        data[r_index,11]=0
        data[r_index,12]=0
        data[r_index,13]=0
        data[r_index,14]=0
    if EE_data['Glazing Area Distribution'][r_index]==1 :   
        data[r_index,10]=1
        data[r_index,11]=0
        data[r_index,12]=0
        data[r_index,13]=0
        data[r_index,14]=0    
    if EE_data['Glazing Area Distribution'][r_index]==2 :   
        data[r_index,10]=0
        data[r_index,11]=1
        data[r_index,12]=0
        data[r_index,13]=0
        data[r_index,14]=0    
    if EE_data['Glazing Area Distribution'][r_index]==3 :   
        data[r_index,10]=0
        data[r_index,11]=0
        data[r_index,12]=1
        data[r_index,13]=0
        data[r_index,14]=0    
    if EE_data['Glazing Area Distribution'][r_index]==4 :   
        data[r_index,10]=0
        data[r_index,11]=0
        data[r_index,12]=0
        data[r_index,13]=1
        data[r_index,14]=0    
    if EE_data['Glazing Area Distribution'][r_index]==5 :   
        data[r_index,10]=0
        data[r_index,11]=0
        data[r_index,12]=0
        data[r_index,13]=0
        data[r_index,14]=1           
    data[r_index,15]=EE_data['Cooling Load'][r_index]#/42.96
    data[r_index,16]=EE_data['Heating Load'][r_index]      
random.shuffle(data)
Train_data[0:rows*75//100,0:17]=data[0:rows*75//100,0:17]
Test_data[0:rows*25//100,0:17]=data[rows*75//100:rows,0:17]
###########################################################################################
for t in range(Training_time):
###########################################################################################    
    for i in range(0,train):
#    for i in range(0, 50):
        inp=Train_data[i,0:16].T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=Train_data[i,16]
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
plt.figure ( figsize = ( 20 , 10 )) 
plt.plot ( train_layer_4 , label = "$output$" , color = "red" , linewidth = 1 ) 
plt.plot ( train_y , "b--" , label = "$label$" , linewidth = 0.3) 
plt.xlabel ( "Time(s)" ) 
plt.ylabel( "Value" ) 
plt.title ( "Training" ) 
plt.legend () 
plt.show ()
###########################################################################################
for i in range(0,test):
#    for i in range(0, 50):
        inp=Test_data[i,0:16].T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=Test_data[i,16]
        # Compute and print loss
        test_loss = np.square(layer_4 - y).sum() # loss function
        test_loss_col.append(test_loss)
        test_y.append(y)
        test_layer_4.append(layer_4)
    #    print(i, loss, layer_4)
plt.plot(test_loss_col)
plt.show()
###########################################################################################
plt.figure ( figsize = ( 20 , 10 )) 
plt.plot ( test_layer_4 , label = "$output$" , color = "red" , linewidth = 2 ) 
plt.plot ( test_y , "b" , label = "$label$" , linewidth = 1) 
plt.xlabel ( "Time(s)" ) 
plt.ylabel( "Value" ) 
plt.title ( "Testing" ) 
plt.legend () 
plt.show ()
###########################################################################################
End = time.time() 
print("Total %f sec" % (End - Start)) 