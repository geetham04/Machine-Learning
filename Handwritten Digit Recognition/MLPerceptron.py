# Geetha Madhuri Chittuluri
# MACHINE LEARNING Programming Assignment#1
#Two layer neural network with one hidden layer to perform the handwritten digit recognition task 

import sys
import numpy as np
import pandas as pd
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

#initialization
epochs = 50
learning_rate = 0.1
hidden_1=100
hidden_2=20
hidden_3=50
momentum_1 = 0
momentum_2 = 0.25
momentum_3 = 0.5
momentum_4 = 0.9


#extraction of training data

train_images, train_targets = loadlocal_mnist(
        images_path = 'train-images.idx3-ubyte', 
        labels_path = 'train-labels.idx1-ubyte')

#one hot encoding
train_targets_onehotencoded = np.full([60000,10], 0.1, dtype=float)
for row in range(len(train_targets)):
    index = train_targets[row].astype(int)
    train_targets_onehotencoded[row][index] += 0.8
    
#dividing inputs(images) by 255 and adding bias input of 1 to inputs(images) matrix
train_images_matrix = np.copy(train_images)
train_images_matrix = train_images_matrix / np.float32(255)
bias_col = np.ones((60000,1))
train_images_matrix = np.append(train_images_matrix, bias_col,axis = 1)



#extraction of testing data

test_images, test_targets = loadlocal_mnist(
        images_path = 't10k-images.idx3-ubyte', 
        labels_path = 't10k-labels.idx1-ubyte')

#one hot enconding
test_targets_onehotencoded = np.full([10000,10], 0.1, dtype=float)
for row in range(len(test_targets)):
    index = test_targets[row].astype(int)
    test_targets_onehotencoded[row][index] += 0.8

#dividing inputs(images) by 255 and adding bias input of 1 to inputs(images) matrix
test_images_matrix = np.copy(test_images)
test_images_matrix = test_images_matrix / np.float32(255)
bias_col = np.ones((10000,1))
test_images_matrix = np.append(test_images_matrix, bias_col,axis = 1)



#intializing output matrices for training and testing data
train_y = np.zeros((60000,1))
test_y = np.zeros((10000,1))
hj = np.zeros(hidden_1+1)
hj[0]=1
train_accuracy = np.zeros((epochs))
test_accuracy = np.zeros((epochs))

#creating weights
hidden_weights = np.random.uniform(-0.05, 0.05, size=(785, hidden_1))
output_weights = np.random.uniform(-0.05, 0.05, size=(hidden_1+1,10))

updated_input2hidden_weights = np.zeros((hidden_1,785))                                        
updated_hidden2output_weights = np.zeros((hidden_1+1,10))


#Forward phase
def mlpfwd(iter,inputs,hidden_weights,output_weights,targets_onehotencoded):
    # activation function of hidden and output units
    xiwji = np.dot(inputs[iter][:],hidden_weights)
    hj[1:] = 1/(1+(np.exp(-xiwji)))

    hjwkj = np.dot(hj, output_weights)
    ok = 1/(1+(np.exp(-hjwkj)))

    # error terms calculation of output and hidden units 
    error_outputunit = ok * (1-ok) * (targets_onehotencoded[iter][:] - ok)

    x = hj[1:] * (1-hj[1:])
    y = np.dot(output_weights[1:,:], np.transpose(error_outputunit))
    error_hiddenunit = x * y

    index = np.argmax(ok, axis = 0)

    return index, error_outputunit, error_hiddenunit, hj;


#Backward propogation
def mlpbwd(error_outputunit, error_hiddenunit, hj, updated_hidden2output_weights, updated_input2hidden_weights, hidden_weights, output_weights):
    # update weights with momentum

    # hidden to output layer update weights
    a = learning_rate * np.outer(hj,error_outputunit)
    #print(hj.shape, error_outputunit.shape)
    b = updated_hidden2output_weights * momentum_1
    new_updated_hidden2output_weights = a + b

    #input to hidden layer update weights
    c = learning_rate * np.outer(error_hiddenunit,train_images_matrix[row][:])
    #print(train_images_matrix)
    d = updated_input2hidden_weights * momentum_1
    new_updated_input2hidden_weights = c + d

    hidden_weights += np.transpose(new_updated_input2hidden_weights)
    updated_input2hidden_weights = new_updated_input2hidden_weights
    output_weights += new_updated_hidden2output_weights
    updated_hidden2output_weights = new_updated_hidden2output_weights
    
    return updated_hidden2output_weights, updated_input2hidden_weights, hidden_weights, output_weights;


#testing algorithm on trained data

for i in range(epochs):
    for row in range(len(train_images_matrix)):
        index, error_outputunit, error_hiddenunit, hj = mlpfwd(row,train_images_matrix,hidden_weights,output_weights,train_targets_onehotencoded)

        updated_hidden2output_weights, updated_input2hidden_weights, hidden_weights, output_weights = mlpbwd(error_outputunit, error_hiddenunit, hj, updated_hidden2output_weights, updated_input2hidden_weights, hidden_weights, output_weights)

        train_y[row] = index                                

    train_actual = train_y.transpose().reshape(60000,).astype(int)
    train_cm = confusion_matrix(train_actual, train_targets)
    train_diag_sum =  sum(np.diag(train_cm))
    train_accuracy[i] = (train_diag_sum/60000.00)*100


#testing algorithm on test data

    for row in range(len(test_images_matrix)):
        index, error_outputunit, error_hiddenunit, hj = mlpfwd(row,test_images_matrix,hidden_weights,output_weights,test_targets_onehotencoded)

        test_y[row] = index                                
                                

    test_actual = test_y.transpose().reshape(10000,).astype(int)
    test_cm = confusion_matrix(test_actual, test_targets)
    test_diag_sum =  sum(np.diag(test_cm))
    test_accuracy[i] = (test_diag_sum/10000.00)*100                                        

                                        
    
#plots
plt.plot(train_accuracy,label='training accuracy')
plt.plot(test_accuracy,label='testing accuracy')
plt.ylabel("Accuracy in %")
plt.xlabel("Epoch")

image= "Momentum0.png"
plt.title("For 100 hidden units and momentum value of 0")
plt.gca().legend()
plt.savefig(image)
plt.show()

print ("CONFUSION MATRIX OF TRAIN SET : For 100 hidden units and Momentum value of 0 \n")
print (train_cm)

print ("CONFUSION MATRIX OF TEST SET : For 100 hidden units and Momentum value of 0 \n")
print (test_cm)

print ("train_accuracy: ")
print (train_accuracy)

print("test_accuracy:")
print(test_accuracy)
