import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in xrange(num_train):
    # vector with C (num_classes) scalar elements
    scores = X[i].dot(W)
    # numerical problems:
    # first shift the values of f so that the highest number is 0:
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    normalizer = np.sum(scores_exp)
    probabilities = scores_exp / normalizer

    for j in xrange(num_classes):
      if j == y[i]:
        loss += -np.log(probabilities[j])
        dW[:, j] += (probabilities[j] - 1) * X[i]
      else:
        dW[:, j] += (probabilities[j]) * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  #loss += 0.5 * reg * np.sum(W * W)
  #dW += 0.5 * reg * 2 * W
  loss += reg * np.sum(W * W)
  dW += 0.5 * reg * W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  scores -= np.max(scores,axis=1).reshape(scores.shape[0], 1)
  scores_exp = np.exp(scores)
  normalizer = np.sum(scores_exp, axis=1).reshape(scores.shape[0], 1)
  probabilities = (scores_exp / normalizer)

  lossProbs = np.ones_like(probabilities)
  lossProbs[np.arange(lossProbs.shape[0]), y] = probabilities[np.arange(probabilities.shape[0]), y]

  minusOne = np.copy(probabilities)
  minusOne[np.arange(minusOne.shape[0]), y] -= 1

  num_train = X.shape[0]

  loss = np.sum(-np.log(lossProbs))
  dW = np.dot(X.T, minusOne)


  # Right now the loss is a sum over all training examples, but we want it to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 0.5 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

