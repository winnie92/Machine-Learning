
# coding: utf-8

# In[1]:

import math
import numpy as np
from collections import Counter


# In[211]:

#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cnt = Counter(Y)
        s=sum(cnt.values())
        p=np.array(cnt.values()).astype(float)/float(s)
        e=np.sum(-p*np.log2(p))   

     
    
        #########################################
        return e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        sX=sorted(X)
        wx=np.array(Counter(sX).values()).astype(float)/float(sum(Counter(sX).values()))
    #    print Counter(X)
        xy=zip(X,Y)
        xy = sorted(xy,key=lambda x:x[0])
    #    print xy
        initx=sX[0]
        ylist=[]
        for i in range(len(Counter(X))):
            ylist.append([])
        i=0
        for (x,y) in xy:
            if x == initx:
                ylist[i].append(y)
            else:
                i=i+1
                ylist[i].append(y)
            initx=x
        celist=[]
        
        for i in range(len(ylist)):
            celist.append(wx[i]*Tree.entropy(ylist[i]))
        ce=sum(celist)        
    
        #########################################
        return ce 
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y)-Tree.conditional_entropy(Y,X)
 
        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        m=[]
        for i in range(X.shape[0]):
            m.append((i,Tree.information_gain(Y,X[i])))
        z = sorted(m,key=lambda x:x[1], reverse=True)
        i = z[0][0]

 
        #########################################
        return i

    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        keys=X[i,]
        #vx=np.delete(X, i, 0)
        vy=Y.reshape((1,len(Y)))
        v=np.matrix(np.r_[X,vy]).T
        z=zip(keys,v)
        z1=sorted(z,key=lambda x:x[0])

        dkeys=sorted(Counter(zip(*z1)[0]).keys())
        xylist=[]
        for i in range(len(dkeys)):
            xylist.append([])

        i=0
        initk=z1[0][0]
        for (k,v) in z1:
            if k == initk:
                xylist[i].append(v)
            else:
                i=i+1
                xylist[i].append(v)
            initk=k

        ylist=[]
        stacked = []
        for xy in xylist:
            tmp=np.stack(xy).T
            tmp2=np.delete(tmp,-1, 0)
            stacked.append(tmp2)
            ylist.append(np.array(tmp[-1,]).flatten())
    #         print stacked        
    #         print ylist
        C={}
        for i in range(len(dkeys)):
            C[dkeys[i]]=Node(np.asarray(stacked[i]),ylist[i], i=None,C=None, isleaf= False,p=None)

        #########################################
        return C 
    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
#         Y=sorted(Y)
#         if Y[0] == Y[-1]:
#             s = True
#         else:
#             s = False 
        inity=Y[0]
        s=np.all(Y == inity)
        #########################################
        return s
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attributes. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
#         X=np.matrix(X)
        tmp=[]
        for i in range(len(X)):
            tmp.append(Tree.stop1(np.array(X[i]).flatten()))
        s=np.all(np.array(tmp) == True)
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cnt=Counter(Y)
        initc=cnt.values()[0]
        if np.all(initc==cnt.values()):
            y = cnt.values()
        y = cnt.most_common(1)[0][0]
        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf= True
            t.p=Tree.most_common(t.Y)
            return
        # find the best attribute to split
        t.i = Tree.best_attribute(t.X,t.Y)
        # recursively build subtree on each child node
        t.C = Tree.split(t.X,t.Y,t.i)
        t.p = Tree.most_common(t.Y)
        for k,v in t.C.items():  
            Tree.build_tree(v) 
        #########################################

    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X=X,Y=Y)
        Tree.build_tree(t)
 
        #########################################
        return t
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        global y
        if t.isleaf == True or (x[t.i] not in t.C.keys()):
            y = t.p
        else:
            for k,v in t.C.items():
                if x[t.i] == k:
                    y = Tree.inference(v,x)
        #########################################
        return y
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in range(X.shape[1]):
            x = X[:,i]
            Y.append(Tree.inference(t,x))
        Y = np.array(Y)
        #########################################
        return Y

    #--------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
        X = tmp[1:,1:].T#加载数据部分
        Y = tmp[1:,0]#加载类别标签部分
        #########################################
        return X,Y




# In[207]:



