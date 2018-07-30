from problem1 import *
import numpy as np
import sys
'''
    Unit test 1:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (30 points in total)---------------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)

#-------------------------------------------------------------------------
def test_E_step():
    ''' (10 points) E_Step'''
    # 4 data samples, p = 2
    X = np.mat('0.,0.;1.,0.;0.,1;1.,1.')
    mu = np.array([[0.,0.],[1.,1.]])
    sigma = np.array([
                      [# 1st component
                       [1.,0.],
                       [0.,1.]
                      ], 
                      [# 2nd component
                       [1.,0.],
                       [0.,1.]
                      ]  
                     ]
                    )
    PY = np.array([.5,.5])
    Y = E_step(X,mu,sigma,PY) 
    
    assert Y.shape == (4,2)
    Y_true = [[ 0.73105858, 0.26894142],
              [ 0.5       , 0.5       ],
              [ 0.5       , 0.5       ],
              [ 0.26894142, 0.73105858]] 
    assert np.allclose(Y, Y_true, atol = 1e-3)

    # 4 data samples, p = 3, k = 2
    X = np.mat('0.,0.,0.;10.,10.,10.;0.,0.,0.;10.,10.,10.')
    mu = np.array([[0.,0.,0.],[10.,10.,10.]])
    sigma = np.array([
                      [# 1st component
                       [1.,0.,0.],
                       [0.,1.,0.],
                       [0.,0.,1.]
                      ], 
                      [# 2nd component
                       [1.,0.,0.],
                       [0.,1.,0.],
                       [0.,0.,1.]
                      ]  
                     ]
                    )
    PY = np.array([.5,.5])
    Y = E_step(X,mu,sigma,PY) 
    assert Y.shape == (4,2)
    Y_true = [[ 1., 0.],
              [ 0., 1.],
              [ 1., 0.],
              [ 0., 1.]] 
    assert np.allclose(Y, Y_true, atol = 1e-3)
 

#-------------------------------------------------------------------------
def test_M_step():
    ''' (10 points) M_Step'''
    # 4 data samples, p = 2
    X = np.mat('0.,1.;1.,0.;0.,-1;-1.,0.').getA()
    Y = np.array([[ 0.5, 0.5],
                  [ 0.5, 0.5],
                  [ 0.5, 0.5],
                  [ 0.5, 0.5]])
    mu,sigma,PY = M_step(X,Y)
    assert np.allclose(mu,np.zeros((2,2))/2., atol= 0.1)
    assert np.allclose(PY,np.ones(2)/2., atol= 0.1)
    sigma_true = [[[ 0.5, 0. ],
                   [ 0. , 0.5]],
                  [[ 0.5, 0. ],
                   [ 0. , 0.5]]]
    assert np.allclose(sigma, sigma_true,atol = 1e-2)
    Y = np.array([[ 1.0, 0.0],
                  [ 1.0, 0.0],
                  [ 0.0, 1.0],
                  [ 0.0, 1.0]])
    mu,sigma,PY = M_step(X,Y)
    assert np.allclose(mu,[[.5,.5],[-.5,-.5]], atol= 0.1)
    assert np.allclose(PY,np.ones(2)/2., atol= 0.1)
    Y = np.array([[ 1.0, 0.0],
                  [ 1.0, 0.0],
                  [ 1.0, 0.0],
                  [ 0.0, 1.0]])
    mu,sigma,PY = M_step(X,Y)
    assert np.allclose(mu,[[.333,0.],[-1.,0.]], atol= 1e-2)
    assert np.allclose(PY,[.75,.25], atol= 1e-2)


    mu = np.array([[0.,0.],
                   [5.,5.]])
    sigma = [[[ 1., 0.],
              [ 0., 4.]],
             [[ 4., 0.],
              [ 0., 1.]]]
    n = 1000
    X1 = np.random.multivariate_normal(mu[0],sigma[0],n)
    X2 = np.random.multivariate_normal(mu[1],sigma[1],n)
    X = np.concatenate([X1,X2],axis=0)
    Y1 = np.ones((n,1))
    Y0 = np.zeros((n,1))
    Ya = np.concatenate([Y1,Y0],axis=1)
    Yb = np.concatenate([Y0,Y1],axis=1)
    Y = np.concatenate([Ya,Yb],axis=0)
    mu_,sigma_,PY_ = M_step(X,Y) 
    assert np.allclose(mu_,mu,atol= 1.)
    assert np.allclose(PY_,[.5,.5], atol= 1.)
    assert np.allclose(sigma_,sigma, atol= 1.)


#-------------------------------------------------------------------------
def test_EM():
    ''' (10 points) EM'''
    for _ in xrange(3):
        mu = np.array([[0.,0.],[5.,5.]])
        sigma = [[[ 1., 0.],
                  [ 0., 4.]],
                 [[ 4., 0.],
                  [ 0., 1.]]]
        n = 1000
        X1 = np.random.multivariate_normal(mu[0],sigma[0],n)
        X2 = np.random.multivariate_normal(mu[1],sigma[1],n)
        X = np.concatenate([X1,X2],axis=0)
        Y_,mu_,sigma_,PY_ = EM(X,num_iter=20)
        Y1 = np.ones((n,1))
        Y0 = np.zeros((n,1))
        Ya = np.concatenate([Y1,Y0],axis=1)
        Yb = np.concatenate([Y0,Y1],axis=1)
        Y = np.concatenate([Ya,Yb],axis=0)
        print 'mu:',mu_
        print 'sigma:',sigma_
        print 'PY:',PY_
        print 'Y:',Y_
        assert np.allclose(mu_,mu, atol= 1.) or np.allclose(mu_,mu[::-1], atol= 1.)
        assert np.allclose(PY_,[.5,.5], atol= 1.)
        assert np.allclose(Y_,Y, atol= 1.) or np.allclose(Y_,Y[::-1], atol= 1.)

    assert np.allclose(sigma_,sigma, atol= 1.) or np.allclose(sigma_,sigma[::-1], atol= 1.)



