
# coding: utf-8

# In[1]:

import math
import numpy as np


# In[16]:

def predict(X, w, b):
    t=X*w+b
    y=[]
    for e in t:
        if e>=0:
            y.append(1)
        else:
            y.append(-1)
    y=np.asmatrix(y).reshape((len(y),1))
    return y 


# In[17]:




# In[25]:

def subgradient(x, y, w, b, l=0.001):
    '''
        Compute the subgradient of loss function w.r.t. w and b (on one training instance).
        Input:
            x: the feature vector of a training data instance, a numpy vector of shape p by 1
               Here p is the number of features
            y: the label of the training data instance, a float scalar (1. or -1.) 
            w: the current weights of the SVM model, a numpy float vector of shape p by 1. 
            b: the current bias of the SVM model, a float scalar.
            l: (lambda) = 1/ (n C), which is the weight of the L2 regularization term. 
                Here n is the number of training instances, C is the weight of the hinge loss.
        Output:
            dL_dw : the subgradient of the weights, a numpy float vector of shape p by 1.
                The i-th element is  d L / d w[i] 
            dL_db : the sbugradient of the bias, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    t=1-y*(x.T*w+b)
    if t>0:
        dL_dw=l*w-y*x
        dL_db=(-1)*y
    else:
        dL_dw=l*w
        dL_db=0      
    #########################################
    return dL_dw, dL_db 



# In[26]:




# In[27]:

#--------------------------
def update_w(w, dL_dw, lr=0.01):
    '''
        Update the parameter w using the subgradient.
        Input:
            w: the current weights of the SVM model, a numpy float vector of shape p by 1. 
            dL_dw : the subgradient of the weights, a numpy float vector of shape p by 1.
                The i-th element is  d L / d w[i] 
            lr: the learning rate, a float scalar, controling the speed of gradient descent.
        Output:
            w: the updated weights of the SVM model, a numpy float vector of shape p by 1. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    w=w-lr*dL_dw

    #########################################
    return w 


# In[28]:




# In[29]:

#--------------------------
def update_b(b, dL_db, lr=0.01):
    '''
        Update the parameter b using the subgradient.
        Input:
             b: the current weights of the SVM model, a float scalar.
            dL_db : the subgradient of the weights, a numpy float vector of shape p by 1.
            lr: the learning rate, a float scalar, controling the speed of gradient descent.
        Output:
            b: the updated bias of the SVM model, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    b=b-lr*dL_db



    #########################################
    return b




# In[30]:




# In[32]:

#--------------------------
def train(X, Y, lr=0.01,C = 1., n_epoch = 10):
    '''
        Train the SVM model using Stochastic Gradient Descent (SGD).
        Input:
            X: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            Y : the sample labels, a numpy vector of shape n by 1.
            lr: the learning rate, a float scalar, controling the speed of gradient descent.
            C: the weight of the hinge loss, a float scalar.
            n_epoch: the number of rounds to go through the instances in the training set.
        Output:
            w: the weights of the SVM model, a numpy float vector of shape p by 1. 
            b: the bias of the SVM model, a float scalar.
    '''
    n,p = X.shape

    #l: (lambda) = 1/ (n C), which is the weight of the L2 regularization term. 
    l = 1./(n * C)

    w,b = np.asmatrix(np.zeros((p,1))), 0. # initialize the weight vector as all zeros
    for _ in xrange(n_epoch):
        for i in xrange(n):
            x = X[i].T # get the i-th instance in the dataset
            y = float(Y[i]) 
            #########################################
            ## INSERT YOUR CODE HERE
            w=update_w(w,subgradient(x, y, w, b, l)[0],lr)
            b=update_b(b,subgradient(x, y, w, b, l)[1],lr)

            #########################################
    return w,b



# In[33]:

'''
    # an example feature matrix (4 instances, 2 features)
    X  = np.mat( [[0., 0.],
                  [1., 1.]])
    Y = np.mat([-1., 1.]).T
    w, b = train(X, Y, 0.01, n_epoch = 1000)
    assert np.allclose(w[0]+w[1]+ b, 1.,atol = 0.1)  # x2 a positive support vector 
    assert np.allclose(b, -1.,atol =0.1)  # x1 a negative support vector 

    #------------------
    # another example
    X  = np.mat( [[0., 1.],
                  [1., 0.],
                  [2., 0.],
                  [0., 2.]])
    Y = np.mat([-1., -1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 10000., n_epoch = 1000)
    assert np.allclose(w[0]+b, -1, atol = 0.1)
    assert np.allclose(w[1]+b, -1, atol = 0.1)
    assert np.allclose(w[0]+w[1]+b, 1, atol = 0.1)
 

    w, b = train(X, Y, 0.01, C= 0.01, n_epoch = 1000)
    assert np.allclose(w, [0,0], atol = 0.1)

    #------------------
    X  = np.mat( [[0., 0.],
                  [1., 1.],
                  [0., 10]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('1;1'), atol = 0.1)

    #------------------
    X  = np.mat( [[0., 0.],
                  [2., 2.],
                  [0., 190]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.5;.5'), atol = 0.1)


    #------------------
    X  = np.mat( [[0., 0.],
                  [5., 5.],
                  [0., 190]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.2;.2'), atol = 0.1)


    #------------------
    X  = np.mat( [[0., 0.],
                  [1., 1.],
                  [0., 1.5]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.68;1.34'), atol = 0.1)

    #------------------
    X  = np.mat( [[ 10., 0.],
                  [ 0., 10.],
                  [-10., 0.],
                  [ 0.,-10.]])
    Y = np.mat([ 1., 1.,-1.,-1.]).T
    w, b = train(X, Y, 0.001, C= 1e10, n_epoch = 1000)
    assert np.allclose(b, 0, atol = 0.1)
    assert np.allclose(w, np.mat('.1;.1'), atol = 0.1)

    #------------------
    X  = np.mat( [[ 15., 0.],
                  [ 0., 10.],
                  [-15., 0.],
                  [ 0.,-10.]])
    Y = np.mat([ 1., 1.,-1.,-1.]).T
    w, b = train(X, Y, 0.001, C= 1e10, n_epoch = 1000)
    print 'w:',w
    print 'b:',b
    assert np.allclose(b, 0, atol = 0.1)
    assert np.allclose(w, np.mat('.1;.1'), atol = 0.1)
''' 

