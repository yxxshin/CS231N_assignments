from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
      exp_scores = np.exp(X[i].dot(W))
      normalized_scores = exp_scores / sum(exp_scores)
      loss += -np.log(normalized_scores[y[i]])
      for j in range(C):
        if j == y[i]:
          dW[:, j] += (normalized_scores[y[i]] - 1) * X[i]
        else:
          dW[:, j] += normalized_scores[j] * X[i]
        

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2 * reg * W
      
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0] 
    C = W.shape[1]

    exp_scores = np.exp(np.dot(X, W))
    scores_row_sum = np.sum(exp_scores, axis = 1)
    normalized_scores = exp_scores / scores_row_sum[:, None]
    loss = np.sum(-np.log(normalized_scores[np.arange(N), y]))
    
    normalized_scores[np.arange(N), y] -= 1
    dW = np.dot(X.T, normalized_scores)
    
    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
