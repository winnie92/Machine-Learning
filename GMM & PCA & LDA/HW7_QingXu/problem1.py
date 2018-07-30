
# coding: utf-8

# In[2]:


import numpy as np
from scipy.stats import multivariate_normal
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: EM algorithm for Gaussian Mixture Model (GMM).
    In this problem, you will implement the expectation-maximization problem of a Gaussian Mixture Distribution.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''


# In[150]:


#--------------------------
def E_step(X,mu,sigma,PY):
    '''
        E-step: Given the current estimate of model parameters, compute the expected mixture of components on each data point. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Hint: you could use multivariate_normal.pdf() to compute the density funciont of Gaussian ditribution.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Y = []
    for x in X:# for each data instance
        tmp = []
        for u,c,p in zip(mu,sigma,PY):
#             print "u shape",u.shape
            tmp.append(np.array(p)*np.array(multivariate_normal.pdf(x,mean=u,cov=c)))
        tmp2=[]
        for e in tmp:
            tmp2.append(e/sum(tmp))
        Y.append(tmp2)
#     print "Y",Y
    Y = np.array(Y)
#     print "Y shape",Y.shape
    
    #########################################
    return Y 
#--------------------------
def M_step(X,Y):
    '''
        M-step: Given the current estimate of label distribution, update the parameters of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Output:
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    mu = []
    sigma = []
    PY = []
    n = X.shape[0]
    p = X.shape[1]
    k = Y.shape[1]
    for j in range(k): #each cluster: A,B,C
        y = Y[:,j]# nx1 --A cluster 的权重 for each data instance
        tmp1 = np.multiply(y.reshape(n,1),X) # nxp
#         tmp1=np.array(y)*np.array(X)
#         print type(np.array(y).reshape(n,1))
        tmp2 = np.array(tmp1.sum(axis=0)/sum(y)).flatten() #1xp
#         print "tmp2",type(tmp2),tmp2.shape
        mu.append(tmp2) #append k 个tmp2: kxp
        
        
        tmp4=0
        for i in range(n):
#             print "t",np.matrix((X[i]-tmp2)).T*np.matrix((X[i]-tmp2))
#             print "T",np.matrix((X[i]-tmp2)).T.shape
#             print "notT",np.matrix((X[i]-tmp2)).shape
#             print "y",type(Y[i,j])
            tmp3 = Y[i,j]*(np.matrix((X[i]-tmp2)).T*np.matrix((X[i]-tmp2)))
            # px1*1xp = pxp for each data instance
#             print "instance",i
#             print "tmp3",tmp3
            tmp4 = tmp4+tmp3 
        tmp5 = tmp4/sum(y) #pxp
        sigma.append(tmp5) # kxpxp
    
        PY.append(sum(y)/n)

    mu=np.array(mu) 
#     print "mu",type(mu),mu
#     print "mu shape",mu.shape
#     print "u shape",mu[0].shape
#     print "mu",mu
    PY = np.array(PY)
    #########################################
    return mu,sigma,PY

#--------------------------
def EM(X,k=2,num_iter=10):
    '''
        EM: Given a set of data samples, estimate the parameters and label assignments of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            k: the number of components in Gaussian mixture, an integer scalar.
            num_iter: the number EM iterations, an integer scalar.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    # initialization (for testing purpose, we use first k samples as the initial value of mu) 
    n,p = X.shape
    mu = X[:k]

    sigma = np.zeros((k,p,p))
    for i in range(k):
        sigma[i] = np.eye(p)
    PY = np.ones(k)/k

    #########################################
    ## INSERT YOUR CODE HERE
    for _ in range(num_iter):
        Y = E_step(X,mu,sigma,PY)

        mu,sigma,PY = M_step(X,Y)
#         print "mu",mu
#         print mu.shape
    #########################################
    return Y,mu,sigma,PY


