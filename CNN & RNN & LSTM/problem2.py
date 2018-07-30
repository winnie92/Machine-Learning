
# coding: utf-8

# In[ ]:


from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD

#-------------------------------------------------------------------------
'''
    Problem 2: Convolutional Neural Network 
    In this problem, you will implement a convolutional neural network with a convolution layer and a max pooling layer.
    The goal of this problem is to learn the details of convolutional neural network. 
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
    Note: please do NOT use th.nn.functional.conv2d or th.nn.Conv2D, implement your own version of 2d convolution using only basic tensor operations.
'''

#--------------------------
def conv2d(x,W,b):
    '''
        Compute the 2D convolution with one filter on one image, (assuming stride=1).
        Input:
            x:  one training instance, a float torch Tensor of shape l by h by w. 
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape l by s by s. 
            b: the bias vector of the convolutional filter, a torch scalar Variable. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (h-s+1) by (w-s+1)
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
#     s=W.size(1)
#     z=[]
#     for lx,lw in x,W:
#         lx=lx.data
#         lw=lw.data
#         zi=[]
#         for i in range(lx.shape[0]):
#             zj=[]
#             starti=i
#             endi=i+s
#             for j in range(lx.shape[1]):
#                 startj=j
#                 endj=j+s
#                 xij=lx[starti:endi,startj:endj]                
#                 zj.append(xij*lw+b)
#                 if endj == lx.shape[1]:
#                     break
#             zi.append(zj)
# #             print zi
#             if endi == lx.shape[0]:
#                 break
#         z.append(zi)
#     z=th.Tensor(z)        
    # num filters K, filter size F, stride S, 
    X = x
    H1 = X[0].size(0)
    W1 = X[0].size(1)
    num_chnl = X.size(0)
    S = 1
    F = W.size(1)
    K = 1
    # get output dimension from filtered image 
    H2 = (H1 - F)/S + 1
    W2 = (W1 - F)/S + 1

    # get row and col representation of W and X, to apply matrix multiplication. 
    # note: row size and col size are same.
    # im2col size is F * F * channel x H2 * W2
    row_col_size = F * F * num_chnl
    W_row = None
    X_col = None


    W_row = W.view(1, row_col_size)
#     print type(W_row)
    rowIdx = 0
    while rowIdx + F <= H1:
        colIdx = 0
        while colIdx + F <= W1:
            filtered = X[:num_chnl, rowIdx: rowIdx + F , colIdx: colIdx + F].contiguous().view(row_col_size, 1)
            if X_col is None:
                X_col = filtered
            else:
                X_col = th.cat([X_col, filtered], 1)
            colIdx += S
        rowIdx += S
#     print X_col
    z = th.mm(W_row, X_col).view(H2, W2) + b
    return z


#--------------------------
def Conv2D(x,W,b):
    '''
        Compute the 2D convolution with multiple filters on a batch of images, (assuming stride=1).
        Input:
            x:  a batch of training instances, a float torch Tensor of shape (n by l by h by w). 
            n is the number instances in a batch.
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape (n_filters by l by s by s). 
            b: the bias vector of the convolutional filter, a torch vector Variable of length n_filters. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by (h-s+1) by (w-s+1) )
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    z = None
    for bx in x:
        tz = None
        for uw, ub in zip(W, b):
            tmp = conv2d(bx, uw, ub)
            tmp.unsqueeze_(0)
            if tz is None:
                tz = tmp
            else:
                tz = th.cat((tz, tmp), 0)
        tz.unsqueeze_(0)
        if z is None:
            z = tz
        else:
            z = th.cat((z, tz), 0)
    return z




    #########################################


#--------------------------
def ReLU(z):
    '''
        Compute ReLU activation. 
        Input:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by h by w )
                h and w are the height and width of the image after convolution. 
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.functional.relu, implement your own version using only basic tensor operations. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    a = th.clamp(z,min=0)

    #########################################
    return a 

#--------------------------
def avgpooling(a):
    '''
        Compute the 2D average pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.AvgPool2d or torch.nn.functional.avg_pool2d or avg_pool1d, implement your own version using only basic tensor operations.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    F=2
    n=a.size(0)
    n_filter=a.size(1)
    h=a.size(2)
    w=a.size(3)
    p= None
    for nn in range(n):
        filters=a[nn] # the filter layers in each instance 
        tmpp = None
        for f in filters: # each filter f is hxw
            rowIdx = 0
            tmp = None
            while rowIdx + F <= h: # each row
                colIdx = 0
                while colIdx + F <= w:
                    avg = f[rowIdx: rowIdx + F,colIdx: colIdx + F].mean()
                    if tmp is None:
                        tmp = avg
                    else:
                        tmp = th.cat([tmp, avg], 0)
                    colIdx += F
                rowIdx += F
            tmp = tmp.view(h / 2, w / 2)
            tmp.unsqueeze_(0)
            if tmpp is None:
                tmpp = tmp
            else:
                tmpp = th.cat((tmpp, tmp), 0)
        tmpp.unsqueeze_(0)
        if p is None:
            p = tmpp
        else:
            p = th.cat((p, tmpp), 0)
    return p
#--------------------------
def maxpooling(a):
    '''
        Compute the 2D max pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after max pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.MaxPool2d or torch.nn.functional.max_pool2d or max_pool1d, implement your own version using only basic tensor operations.
        Note: if there are mulitple max values, select the one with the smallest index.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    F=2
    n=a.size(0)
    n_filter=a.size(1)
    h=a.size(2)
    w=a.size(3)
    p= None
    for nn in range(n):
        filters=a[nn] # the filter layers in each instance 
        tmpp = None
        for f in filters: # each filter f is hxw
            rowIdx = 0
            tmp = None
            while rowIdx + F <= h: # each row
                colIdx = 0
                while colIdx + F <= w:
                    avg = None
                    choosen = f[rowIdx: rowIdx + F,colIdx: colIdx + F]
                    target = choosen.max().data
                    for can in choosen:
                        for var in can:
#                             print var.data, target
                            if (var.data == target).any():
                                avg = var
                                break;
                        if avg is not None:
                            break;
                    if tmp is None:
                        tmp = avg
                    else:
                        tmp = th.cat([tmp, avg], 0)
                    colIdx += F
                rowIdx += F
            tmp = tmp.view(h / 2, w / 2)
            tmp = tmp.unsqueeze(0)
            if tmpp is None:
                tmpp = tmp
            else:
                tmpp = th.cat((tmpp, tmp), 0)
        tmpp = tmpp.unsqueeze(0)
        if p is None:
            p = tmpp
        else:
            p = th.cat((p, tmpp), 0)
#         p = p.view(n, n_filter, h / 2, w / 2)
    return p


#--------------------------
def num_flat_features(h=28, w=28, s=3, n_filters=10):
    ''' Compute the number of flat features after convolution and pooling. Here we assume the stride of convolution is 1, the size of pooling kernel is 2 by 2, no padding. 
        Inputs:
            h: the hight of the input image, an integer scalar
            w: the width of the input image, an integer scalar
            s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
            n_filters: the number of convolutional filters, an integer scalar
        Outputs:
            p: the number of features we will have on each instance after convolution, pooling, and flattening, an integer scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    hout = (h - s) + 1
    wout = (w - s) + 1
    hhout = (hout - 2)/2 + 1
    wwout = (wout - 2)/2 + 1
    p = hhout * wwout * n_filters

    #########################################
    return p

#-------------------------------------------------------
class CNN(sr):
    '''CNN is a convolutional neural network with a convolution layer (with ReLU activation), a max pooling layer and a fully connected layer.
       In the convolutional layer, we will use ReLU as the activation function. 
       After the convolutional layer, we apply a 2 by 2 max pooling layer, before feeding into the fully connected layer.
    '''
    # ----------------------------------------------
    def __init__(self, l=1, h=28, w=28, s=5, n_filters=5, c=10):
        ''' Initialize the model. Create parameters of convolutional layer and fully connected layer. 
            Inputs:
                l: the number of channels in the input image, an integer scalar
                h: the hight of the input image, an integer scalar
                w: the width of the input image, an integer scalar
                s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
                n_filters: the number of convolutional filters, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.conv_W: the weight matrix of the convolutional filters, a torch Variable of shape n_filters by l by s by s, initialized as all-zeros. 
                self.conv_b: the bias vector of the convolutional filters, a torch vector Variable of length n_filters, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer, a torch Variable of shape (p, c), initialized as all-zeros. 
                        Hint: CNN is a subclass of SoftmaxRegression, which already has a W parameter. p is the number of flat features after pooling layer.
                self.b: the bias vector parameter, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            Note: In this problem, the parameters are initialized as either all-zeros or all-ones for testing purpose only. In real-world cases, we usually initialize them with random values.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # compute the number of flat features
         
        # initialize fully connected layer 

        # the kernel matrix of convolutional layer 
        p = num_flat_features(h, w, s, n_filters)
        super(CNN, self).__init__(p, c)
        # compute the number of flat features
        
        self.conv_W = Variable(th.zeros(n_filters, l, s, s), requires_grad = True)
        self.conv_b = Variable(th.ones(n_filters), requires_grad = True)
        self.p = p


        #########################################


    # ----------------------------------------------
    def forward(self, x):
        '''
           Given a batch of training instances, compute the linear logits of the outputs. 
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by l by h by w. 
                Here n is the batch size. l is the number of channels. h and w are height and width of an image. 
            Output:
                z: the logit values of the batch of training instances after the fully connected layer, 
                a float matrix of shape n by c. Here c is the number of classes
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        # convolutional layer
        n = x.size(0)
#         print "before conv2d", x.size()
        tmp = Conv2D(x, self.conv_W, self.conv_b)
#         print "after conv2d" ,tmp.size()
        # ReLU activation 
        tmp = ReLU(tmp)
#         print "after relu " ,tmp.size()
        # maxpooling layer
        tmp = maxpooling(tmp)
        # flatten 
#         print ("after maxpooling", tmp.size())
#         tmp = tmp.view(n, tmp.size(1) * tmp.size(2) * tmp.size(3)) ##########################
        tmp = tmp.view(n, self.p)
        # fully connected layer
        z = super(CNN, self).forward(tmp)
        #########################################
        return z

    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                n_steps: the number of batches of data to train, an integer scalar
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
        """
        # create a SGD optimizer
        optimizer = SGD([self.conv_W,self.conv_b,self.W,self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                #########################################
                ## INSERT YOUR CODE HERE

                # forward pass
                z = self.forward(x)
                # compute loss 
                L = super(CNN, self).compute_L(z, y)
                # backward pass: compute gradients
                sr.backward(self, L)
                # update model parameters
                optimizer.step()
                # reset the gradients 
                optimizer.zero_grad()
                #########################################
                count+=1
                if count >=n_steps:
                    return 


