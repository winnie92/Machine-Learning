from problem3 import *
import numpy as np
import sys

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (40 points in total)---------------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)


#-------------------------------------------------------------------------
def test_variational_inference():
    '''(10 points) variational_inference'''
    # a document with 5 words ( 0, 1, 2 denote the IDs of the words in the vocabulary)
    w = np.array([0, 0, 1, 1, 2])
    p=3
    k=2
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=0)
    assert np.allclose(gamma,[3.5,3.5],atol=1e-1)
    assert np.allclose(phi,np.ones((5,2))*.5,atol=1e-1)
       
    p=4
    k=3
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=2., n_iter=0) 
    assert np.allclose(gamma,3.667*np.ones(3),atol=1e-2)
    assert np.allclose(phi,np.ones((5,3))*.33,atol=1e-2)

    p=3
    k=2
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=1)
    assert np.allclose(gamma,3.5*np.ones(2),atol=1e-2)
    assert np.allclose(phi,np.ones((5,2))*.5,atol=1e-2)


    beta=np.array([[.2,.8], # topic 0
                   [.3,.7], # topic 1
                  ])

    w = np.array([0, 0, 1, 1])
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=1)
    assert np.allclose(gamma,[2.86666667, 3.13333333],atol=1e-2)
    phi_true = [[ 0.4       , 0.6       ],
                [ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
    assert np.allclose(phi,phi_true,atol=1e-2)

    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=30)
    assert np.allclose(gamma,[ 2.44057924, 3.55942076],atol=1e-2)
    phi_true = [[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]]
    assert np.allclose(phi,phi_true,atol=1e-2)


    w = np.array([0, 1, 1, 1])
    gamma, phi = variational_inference(w,beta,alpha=2., n_iter=10)
    assert np.allclose(gamma,[4,4],atol=1e-2)
    phi_true = [[ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
    assert np.allclose(phi,phi_true,atol=1e-2)


#-------------------------------------------------------------------------
def test_E_step():
    '''(10 points) E_step'''

    # Document (m = 1, n = 4)
    W = np.array([
                    [0, 0, 1, 1] # 0-th document: word 0, word 1, ...
                 ])
    beta=np.array([[.2,.8], # topic 0 (word 0, word 1)
                   [.3,.7], # topic 1 (word 0, word 1)
                  ])

    gamma, phi = E_step(W,beta,alpha=1., n_iter=30)
    assert np.allclose(gamma.shape, [1,2])
    assert np.allclose(phi.shape, [1,4,2])
    assert np.allclose(gamma,[[2.44057924, 3.55942076]],atol=1e-2)
    phi_true =[[[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]]]
    assert np.allclose(phi,phi_true,atol=1e-2)


    # Document (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 1, 1, 1], # 1-st document: word 0, word 1, ...
    ])
    gamma, phi = E_step(W,beta,alpha=1., n_iter=30)
    print 'gamma:', gamma
    print 'phi:', phi
    assert np.allclose(gamma.shape, [2,2])
    assert np.allclose(phi.shape, [2,4,2])
    assert np.allclose(gamma,[[2.44057924, 3.55942076],[3,3]],atol=1e-2)
    phi_true =[# document 0
               [[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]],
               # document 1
               [[ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
              ]
    assert np.allclose(phi,phi_true,atol=1e-2)


#-------------------------------------------------------------------------
def test_udpate_beta():
    '''(10 points) update_beta'''
    # 2 Documents, each with 4 words (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 0, 1, 1], # 1-st document: word 0, word 1, ...
    ]) 
    # 2 topics 
    phi = np.array(
    [
      [ # document 0
        [.1,.9],#  word 0 
        [.2,.8],#  word 1 
        [.3,.7], 
        [.4,.6] 
      ],
      [ # document 1
        [.1,.9],#  word 0 
        [.2,.8],#  word 1 
        [.3,.7], 
        [.4,.6] 
      ]
    ]) 
    beta = update_beta(W,phi,2)
    assert np.allclose(beta,[[.3,.7],[.567,.433]],atol=1e-2)

    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 0, 2, 2], # 1-st document: word 0, word 1, ...
    ]) 

    beta = update_beta(W,phi,3)
    print beta
    assert np.allclose(beta,[[.3,.35,.35],[.567,.217,.217]],atol=1e-2)




#-------------------------------------------------------------------------
def test_EM():
    '''(10 points) EM'''
    # 2 Documents, each with 4 words (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 1, 1, 1], # 1-st document: word 0, word 1, ...
    ]) 
    beta, gamma, phi = EM(W, k=2, p=2,n_iter_em=1)
    

    
    beta_true =  [[ 0.31716517, 0.68283483],
                  [ 0.43351708, 0.56648292]]
    gamma_true = [[ 2.78169736, 3.21830264],
                  [ 3.24175736, 2.75824264]]
    phi_true = [[[ 0.39552199, 0.60447801],
                 [ 0.39552199, 0.60447801],
                 [ 0.49532669, 0.50467331],
                 [ 0.49532669, 0.50467331]],
                [[ 0.48505572, 0.51494428],
                 [ 0.58556721, 0.41443279],
                 [ 0.58556721, 0.41443279],
                 [ 0.58556721, 0.41443279]]]
    assert np.allclose(beta,beta_true,atol=1e-1)
    assert np.allclose(gamma,gamma_true,atol=1e-1)
    assert np.allclose(phi,phi_true,atol=1e-1)

    # 2 Documents, each with 4 words (m = 2, n = 4,p=3)
    W = np.array([
        [0, 0, 1, 2], # 0-th document: word 0, word 1, ...
        [0, 1, 2, 1], # 1-st document: word 0, word 1, ...
    ]) 
    beta, gamma, phi = EM(W, k=2, p=3,n_iter_em=10)
    print 'beta:',beta
    print 'gamma',gamma
    print 'phi',phi


    beta_true =  [[ 0.15400768, 0.57021904, 0.27577328],
                  [ 0.59824692, 0.17778931, 0.22396378]]
    gamma_true = [[ 2.3364669 , 3.6635331 ],
                  [ 3.68383387, 2.31616613]]
    phi_true = [[[ 0.14341669, 0.85658331],
                 [ 0.14341669, 0.85658331],
                 [ 0.62651687, 0.37348313],
                 [ 0.42311665, 0.57688335]],
                [[ 0.33232384, 0.66767616],
                 [ 0.83296758, 0.16703242],
                 [ 0.68557488, 0.31442512],
                 [ 0.83296758, 0.16703242]]]

    assert np.allclose(beta,beta_true,atol=1e-1)
    assert np.allclose(gamma,gamma_true,atol=1e-1)
    assert np.allclose(phi,phi_true,atol=1e-1)

    
