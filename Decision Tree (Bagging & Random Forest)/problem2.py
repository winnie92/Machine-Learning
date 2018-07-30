
# coding: utf-8

# In[11]:




# In[182]:

import math
import numpy as np
from problem1 import Tree


# In[183]:

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


# In[353]:

#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #         cp=[]
        #         X=X.astype(float)
        #         z=zip(X,Y)
        #         sz = sorted(z,key=lambda x:x[0])
        #         lasty=sz[0][1]
        #         lastx=sz[0][0]
        #         flag=False
        #         for i in range(1,len(sz)):                
        #             if sz[i][0]!=lastx:
        #                 if sz[i][1]!=lasty or flag==True:
        #                     cp.append((sz[i][0]+sz[i-1][0])/2)
        #                     lasty=sz[i][1]
        #                     flag=False

        #                 lastx=sz[i][0]

        #             elif sz[i][1]!=lasty:
        #                 lasty=z[i][1]
        #                 flag=True
        #         if flag=True:

        #         if len(cp)==0:
        #             cp.append(-np.inf)
        #         cp=np.asarray(cp).T
        #.flatten()
        z=zip(X,Y)
        sz = sorted(z)
        preCnt = 0
        preAtt = None
        dic = {}
        for tu in sz:
            att = tu[0]
            tar = tu[1]
#             print tar
            key = str(att)
            if dic.get(key) == None:
                dic[key] = []
            if len(dic[key]) == 0 or dic[key][-1] != tar:
                dic[key].append(tar)

        keys = sorted(dic)
#         print dic
        pre = keys[0]
        cp =[]
        for i in range(1,len(keys)):
            key = keys[i]
            preVal = dic.get(pre)
            curVal = dic.get(key)
            if (len(curVal) == 1 and len(preVal) == 1):
#                 print ("pre{}, key{}".format(pre, key))
                if preVal[0] == curVal[0]:
#                     print "equal"
                    pre = key
                    continue
                else:
                    cp.append((float(key) + float(pre)) / 2.0)
            else:
                cp.append((float(key) + float(pre)) / 2.0)
            pre = key
        #         print cp
        if len(cp) == 0:
            cp.append(-np.inf)
            cp=np.asarray(cp).T.flatten()
            return cp
        cp=np.asarray(cp).T.flatten()
#         print cp
        #########################################
        return cp
   
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g=-1
        th=-np.inf
        dt = DT()
        cuts=dt.cutting_points(X,Y)
        # print "cuts"
        # print cuts
        tmp = np.copy(X)
        for cut in cuts:
            if cut == -np.inf:
                break
            tmp[X > cut] = cut + 1
            tmp[X <= cut] = 0.0
            gain = Tree.information_gain(tmp, Y)
            tmp = np.copy(X)
            if gain > g:
                g = gain
                th = cut
        #########################################
        return th,g 
    

    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        x1,x2,y1,y2=[],[],[],[]
        X=np.matrix(X)
        for j in range(X.shape[1]):
            if X[i,j] > th:
                x2.append(X[:,j])
                y2.append(Y[j])
            else:
                x1.append(X[:,j])
#                x1=np.c_(x2,(X[:,j],axis=1))
                y1.append(Y[j])
#         if len(x1)==0:
#             # print "Error x1"
#             # print "th",th
# #            print X[i,]
#         else:
        if len(x1)>0:
            for i in range(len(x1)):
                x1[i]=np.array(x1[i]).flatten()
            x1=np.stack(x1,axis=1)
            y1=np.asarray(y1)
            C1=Node(x1, y1)
        else:
            C1=None
        # if len(x2)==0:
        #     # print "Error x2"
        #     # print "th",th
        # else:
        if len(x2)>0:
            for i in range(len(x2)):
                x2[i]=np.array(x2[i]).flatten()            
            x2=np.stack(x2,axis=1)
            y2=np.asarray(y2)
            C2=Node(x2,y2)
        else:
            C2=None
        #(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        # C1=Node(x1, y1)
        # C2=Node(x2,y2)              
        
        #########################################
        return C1, C2
    
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
#         print "t.X"
#         print t.X
#         print "t.Y"
#         print t.Y
        dt=DT()
        # if Condition 1 or 2 holds, stop recursion 
        # if t==None:
        #     return
#         if Tree.stop1(t.Y) or Tree.stop2(t.X):
#             t.isleaf= True
#             t.p=Tree.most_common(t.Y)
#             return  
        if t.isleaf == True:
            return 
        tmp=np.copy(t.X)
        for i in range(tmp.shape[0]):
            th,g=DT.best_threshold(tmp[i],t.Y)
            if th == -np.inf:
                continue
            tmp[i][tmp[i]>th]=th+1
            tmp[i][tmp[i]<=th]=th

        if Tree.stop1(t.Y) or Tree.stop2(tmp):
            t.isleaf= True
            t.p=Tree.most_common(t.Y)
            return       
        # find the best attribute to split
        t.i,t.th = DT().best_attribute(t.X,t.Y)
        if t.th == -np.inf:
            t.isleaf= True
            t.p=Tree.most_common(t.Y)
            return  
        # recursively build subtree on each child node
        t.C1, t.C2 = DT().split(t.X,t.Y,t.i, t.th)        
        t.p = Tree.most_common(t.Y)
        if t.C1!=None:
            DT().build_tree(t.C1)
        if t.C2!=None:
            DT().build_tree(t.C2)

        #########################################
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
#         print "t.i",t.i
#         print "t.th",t.th
#         print "x[i]",x[t.i]
        global y
        if t.isleaf == True :
            y = t.p
        elif x[t.i]>t.th:
            y = DT.inference(t.C2,x)
            #(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):                            
        elif x[t.i]<=t.th:
            y = DT.inference(t.C1,x)  
#         print "y",y
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
                   Each element can be int/float.
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
            Y.append(DT.inference(t,x))
        Y = np.array(Y)
        #########################################
        return Y
    
    #--------------------------
    def train(self, X, Y):
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
        dt=DT()
        dt.build_tree(t)
    
        #########################################
        return t
    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
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
        X = tmp[1:,1:].T.astype(float)
        Y = tmp[1:,0]

        #########################################
        return X,Y   
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
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
            threshold ,gain= DT.best_threshold(col, Y)
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
        print tmp
        m=[]
        for i in range(tmp.shape[0]):
            m.append((i,Tree.information_gain(Y,tmp[i]),threshes[i]))
        z2 = sorted(m,key=lambda x:x[1], reverse=True)
#        print z2
        #deal with same info gain--choose the largest entrophy
#         min_ig=z[0][1]
#         z1=[]
#         for e in z:
#             if e[1]==min_ig:
#                 z1.append((e[0],Tree.entropy(X[e[0]])))
#         z2=sorted(z1,key=lambda x:x[1], reverse=True)
        
        #deal with same info gain and same entropy
        max_en=z2[0][1]
        z3=[]
        for ee in z2:
            if ee[1]==max_en and ee[2]!=-np.inf:
                z3.append((ee[0]))
        i=min(z3)
#        print X
#        i = Tree.best_attribute(tmp, Y)
#        print i

        #########################################
        return i, threshes[i]




