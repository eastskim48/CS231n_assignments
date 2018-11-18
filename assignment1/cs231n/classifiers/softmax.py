import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  WX=np.exp(np.dot(X,W)) #500*10
  n=np.size(y)
  
  for i in range(0,n): #샘플 크기
      p=np.divide(WX[i],np.sum(WX[i]))
      for j in range(0,np.size(W[0])): #class
          if y[i]==j:
            term=p[j]-1
            loss-=log(p[j])
          else:
            term=p[j]
          for k in range(0,np.size(X[0])): #파라미터
              dW[k,j]+=term*X[i,k]
              dW[k,j]+=reg*W[k,j]

  loss/=n
  loss+=reg*(np.sum(W)**2)
  dW=np.divide(dW,n)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  WX=np.exp(np.dot(X,W))
  n=np.size(y)
  nclass=np.size(WX[0])
  WX/=np.dot(WX,np.ones((nclass,nclass)))
  target=np.zeros_like(WX)
  np.add.at(target,tuple([np.arange(n),y]),1)
  dW=np.dot(X.T,WX-target)/n
  dW+=reg*W
  p=WX[np.arange(n),y] #XW는 확률이 계산된 값
  loss=np.dot(np.log(p),np.full(np.size(p),-1/n))
  loss+=reg*(np.sum(W)**2)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

    

