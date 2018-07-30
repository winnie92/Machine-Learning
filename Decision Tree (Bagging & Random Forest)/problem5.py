
# coding: utf-8

# In[146]:

import math
import numpy as np
from problem2 import DT,Node
from problem1 import Tree
# from collections import Counter


# In[362]:

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        u, indices = np.unique(Y, return_inverse=True)
        p=[]
        for i in range(len(u)):
            pi=0
            for j in range(len(D)):
                if i==indices[j]:
                    pi=pi+D[j]
            if pi==0:
                pi=np.exp(-700)
            p.append(pi)
        p=np.array(p)
        e=np.sum(-p*np.log2(p)) 
        #########################################
        return e 
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
# deal with the discrete x
        ux, indicesx = np.unique(X, return_inverse=True)
        px=[]
        d=[]
        y=[]
        for i in range(len(ux)):
            pi=0
            yi=[]
            di=[]
            for j in range(len(D)):
                # when x have the same value
                if i==indicesx[j]:
                    pi=pi+D[j]
#                     di.append(D[j])
                    yi.append(Y[j])
    #deal with px special
            if pi==0:
                pi=np.exp(-700)
    # deal with the weights in yi
#             cyi=Counter(yi)
#             pyi=np.array(cyi.values()).astype(float)/float(np.sum(cyi.values()))
            for ya in yi:
                di.append(1.0/len(yi))
            px.append(pi)
            y.append(yi)
            d.append(di)
        celist=[]
        for i in range(len(px)):
            celist.append(px[i]*DS.entropy(y[i],d[i]))
        ce=sum(celist)     
        
        #########################################
        return ce
    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y,D)-DS.conditional_entropy(Y,X,D)
        #########################################
        return g
    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g=-1
        th=-np.inf
        cuts=DT().cutting_points(X,Y)
        # print "cuts"
        # print cuts
        tmp = np.copy(X)
        for cut in cuts:
            if cut == -np.inf:
                break
            tmp[X > cut] = cut + 1
            tmp[X <= cut] = 0.0
            gain = DS.information_gain(Y,tmp,D)
            tmp = np.copy(X)
            if gain > g:
                g = gain
                th = cut
        #########################################
        return th,g 
    #-----------------------------------------------------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        idx = 0
        threshes = []
        tmp=np.copy(X)
        for col in tmp:
            threshold ,gain= DS.best_threshold(col,Y,D)
#            print threshold
            if threshold == -np.inf:
#                print "pass"
                idx += 1
                threshes.append(-np.inf)
                continue
            col[col>threshold] = threshold + 1
            col[col<= threshold] = threshold
            tmp[idx] = col
            idx += 1
            threshes.append(threshold)
        #choose the largest info gain
        m=[]
        for i in range(tmp.shape[0]):
            m.append((i,DS.information_gain(Y,tmp[i],D),threshes[i]))
        z2 = sorted(m,key=lambda x:x[1], reverse=True)
        
        #deal with same info gain and same entropy
        max_en=z2[0][1]
        z3=[]
        zinf=[]
        for ee in z2:
            if ee[1]==max_en:
                if ee[2]!=-np.inf:
                    z3.append((ee[0]))
                else:
                    zinf.append((ee[0]))
#                 print "error"
#                 print z2
        if len(z3) !=0:
            i=min(z3)
        if np.all(threshes)==-np.inf:
            i=0
#         else:
#             print z2[0][1]==max_en
#             print zinf
#             i=min(zinf)
        th=threshes[i]
        #########################################
        return i,th
    #--------------------------


    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        u, indices = np.unique(Y, return_inverse=True)
        # print u, indices
        p=[]
        for i in range(len(u)):
            pi=0
            for j in range(len(D)):
                if i==indices[j]:
                    pi=pi+D[j]
            if pi<np.exp(-350):
                pi=np.exp(-700)
            p.append(pi)
        z=zip(u,p)
        z=sorted(z,key=lambda x:x[1], reverse=True)
        y = z[0][0]

        #########################################
        return y

    #--------------------------
#     splitted=False
    def build_tree(self,X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
#global splited
#         change the t.X to do stop2
        t=Node(X=X,Y=Y,i=None,th=None,C1=None, C2=None, isleaf= False,p=None)
        tmp=np.copy(t.X)
        for i in range(tmp.shape[0]):
            th,g=DS.best_threshold(tmp[i],t.Y,D)
            # if th == -np.inf:
            #     continue
            tmp[i][tmp[i]>th]=th+1
            tmp[i][tmp[i]<=th]=th

        if Tree.stop1(t.Y) or Tree.stop2(tmp):
            t.isleaf= True
            t.p=DS.most_common(t.Y,D)
            return t      
        # find the best attribute to split
#         self.X=X
#         self.Y=Y
        t.i,t.th = DS().best_attribute(t.X,t.Y,D)
        if t.th == -np.inf:
            t.isleaf= True
            t.p=DS.most_common(t.Y,D)
            return t 
        # recursively build subtree on each child node
        
        t.p = DS.most_common(t.Y,D)
        # split the D
        D1=[]
        D2=[]
        for j in range(len(X[t.i])):
            if X[t.i][j]>t.th:
                D2.append(D[j])
            else:
                D1.append(D[j])
#         for j in range(len(X[t.i])):
#             if tmp[t.i][j]==t.th+1:
#                 D2.append(D[j])
#             else:
#                 D1.append(D[j])
        D1=np.array(D1)
        D2=np.array(D2)
        t.C1, t.C2 = DT().split(t.X,t.Y,t.i, t.th) 
        t.C1.isleaf=True
        t.C1.p=DS.most_common(t.C1.Y,D1)
        t.C2.isleaf=True
        t.C2.p=DS.most_common(t.C2.Y,D2)        
#         if t.C1!=None :
#             DT().build_tree(t.C1.X,t.C1.Y,D1)
#             splitted=True
#         if t.C2!=None:
#             DT().build_tree(t.C1.X,t.C1.Y,D2)
#             splitted=True

        #########################################
        return t
#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e=0
        for i in range(len(Y)):
            if Y[i]!=Y_[i]:
                e=e+D[i]
        
        #########################################
        return e
    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if e <np.exp(-200):
            e=np.exp(-700)
        elif e==1.0:
            a = -350
            return a 
        a=0.5*np.log((1-e)/e)
        #########################################
        return a
    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        tmp=D.copy()
        for i in range(len(tmp)):
            if Y[i]==Y_[i]:
                tmp[i]=tmp[i]*np.exp(-a)
            else:
                tmp[i]=tmp[i]*np.exp(a)
        s = sum(tmp)
        if sum(tmp)<np.exp(-350):
            s=np.exp(-700)
        D=tmp.astype(float)/float(s)
        #########################################
        return D
    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = DS().build_tree(X,Y,D)
        if t.isleaf == True:
            Y_=np.repeat(t.p,len(Y))
        elif t.C1 !=None and t.C2 !=None:
            Y_=[]
            for j in range(len(Y)):
                if X[t.i][j]>t.th:
                    Y_.append(t.C2.p)
                elif X[t.i][j]<=t.th:
                    Y_.append(t.C1.p)
                else:
                    print "X[t.i][j]",X[t.i][j]
        else:
            print "t.C1"
            print t.C1.isleaf
            print t.C1.X
            print t.C1.Y
            print "t.C2"
            print t.C2.isleaf
            print t.C2.X
            print t.C2.Y            
            
        e = AB.weighted_error_rate(Y,Y_,D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D,a,Y,Y_)    
        #########################################
        return t,a,D
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p=[]
        for t in T:
            p.append(DT.inference(t,x))        
        y=DS.most_common(p,A)
#         global y
#         if t.isleaf == True :
#             y = t.p
#         elif x[t.i]>t.th:
#             y = DT.inference(t.C2,x)
#             #(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):                            
#         elif x[t.i]<=t.th:
#             y = DT.inference(t.C1,x)  
        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in range(X.shape[1]):
            x = X[:,i]
            Y.append(AB.inference(x,T,A))
        Y = np.array(Y)
        #########################################
        return Y      

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        T=[]
        A=[]
        # initialize weight as 1/n
        n=X.shape[1]
        D=np.repeat(1/n,n)
        # iteratively build decision stumps
        for i in range(n_tree):
            t,a,D = AB.step(X,Y,D)
            A.append(a)
            T.append(t)
        A=np.array(A)
        #########################################
        return T,A
       


# In[361]:




# In[344]:

# from problem5 import *
# from problem3 import Bag
# import sys
# import numpy as np


# In[363]:

#     b = AB()
#     X, Y = Bag.load_dataset()
#     n = float(len(Y))
#     # train over half of the dataset
#     T,A = b.train(X[:,::2],Y[::2],5) 
#     # test on the other half
#     Y_predict = AB.predict(X[:,1::2],T,A) 
#     accuracy = sum(Y[1::2]==Y_predict)/n*2. 
#     print 'test accuracy of an AdaBoost ensemble of 5 trees:', accuracy
#     assert accuracy >= .85

#     # train over half of the dataset
#     T,A = b.train(X[:,::2],Y[::2],20) 
#     # test on the other half
#     Y_predict = AB.predict(X[:,1::2],T,A) 
#     accuracy = sum(Y[1::2]==Y_predict)/n*2. 
#     print 'test accuracy of an AdaBoost ensemble of 20 trees:', accuracy
#     assert accuracy >= .98


# In[364]:

#     b = AB()

#     X = np.array([[1.,1.,1.,1.],
#                   [2.,2.,2.,2.],
#                   [3.,3.,3.,3.]])
#     Y = np.array(['good','good','good','good'])
#     T,A = b.train(X,Y,1) 
#     print type(A)==np.ndarray
#     print len(T) == 1
#     t = T[0]
#     print t.isleaf == True
#     print t.p == 'good'
# #     assert type(A)==np.ndarray
# #     assert len(T) == 1
# #     t = T[0]
# #     assert t.isleaf == True
# #     assert t.p == 'good' 
#     print "####################"
#     for _ in xrange(20):
#         n_tree = np.random.randint(1,10)
#         T,A = b.train(X,Y,n_tree) 
#         print n_tree,len(T)
# #        assert len(T) == n_tree
#         for i in xrange(n_tree):
#             t = T[i]
#             print t.isleaf
#             print t.p
#             assert t.isleaf == True
#             assert t.p == 'good'  


# In[ ]:



