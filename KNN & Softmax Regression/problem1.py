
# coding: utf-8

# In[8]:

import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    XtrainT = Xtrain.transpose()
    # vecProd = Xtest * XtrainT
    vecProd = np.dot(Xtest,XtrainT)
    # print(vecProd)
    SqXtest =  Xtest**2
    # print(SqXtest)
    sumSqXtest = np.matrix(np.sum(SqXtest, axis=1))
    sumSqXtestEx = np.tile(sumSqXtest.transpose(), (1, vecProd.shape[1]))
    # print(sumSqXtestEx)

    SqXtrain = Xtrain**2
    sumSqXtrain = np.sum(SqXtrain, axis=1)
    sumSqXtrainEx = np.tile(sumSqXtrain, (vecProd.shape[0], 1))    
    SqED = sumSqXtrainEx + sumSqXtestEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    D = np.asarray(ED)
    #########################################
    return D




# In[104]:

#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    D = compute_distance(Xtrain, Xtest)
    index=list(range(len(Ytrain)))
    Ytest=[]
    for e in D:
        z = zip(index,e)
        z = sorted(z,key=lambda x:x[1])
        kid = map(lambda x:x[0], z)[0:K]
        tmp=[]
        for i in kid:
            tmp.append(Ytrain[i])
        print "e= ",e,"tmp= ",tmp
        cnt = Counter(tmp)
        print "cnt= ",cnt
        mode = cnt.get(max(cnt.values()))
        if mode == None:
            mode = tmp[0]
        print "mode= ",mode
        Ytest.append(mode)
    Ytest = np.asarray(Ytest)

    #########################################
    return Ytest 


