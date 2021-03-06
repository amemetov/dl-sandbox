import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  # TODO: Implement the affine forward pass. Store the result in out. You  will need to reshape the input into rows.
  N = x.shape[0]
  D = w.shape[0]
  #M = w.shape[1] # number of neurons
  x_input = x.reshape(N, D)

  z = np.dot(x_input, w) + b
  out = z

  #print 'z.shape', z.shape

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  #dx, dw, db = None, None, None
  # Implement the affine backward pass.

  N = x.shape[0]
  D = w.shape[0]
  # M = w.shape[1] # number of neurons

  x_input = x.reshape(N, D)

  dx = np.dot(dout, w.T)
  dx = dx.reshape(x.shape)

  dw = np.dot(x_input.T, dout)

  db = np.sum(dout, axis=0)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(x, 0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache

  reluD = np.zeros_like(x)
  reluD[x > 0] = 1

  dx = dout * reluD
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # pass

    #1 - compute mean
    sample_mean = x.mean(axis=0)

    #2 - compute variance
    x_minus_mean = x - sample_mean
    x_minus_mean_squared = x_minus_mean**2
    sample_var = 1. / N * np.sum(x_minus_mean_squared, axis=0)

    #3 - normalize
    sqrt_var = np.sqrt(sample_var + eps)
    i_sqrt_var = 1. / sqrt_var
    x_hat = x_minus_mean * i_sqrt_var

    #4 - scale and shift
    y = gamma * x_hat + beta

    # sample_var = x.var(axis=0)
    # x_norm = (x - sample_mean) / np.sqrt(sample_var + eps) # normalize
    # y = gamma * x_norm + beta # scale and shift

    out = y

    cache = {}
    cache['x'] = x
    cache['mean'] = sample_mean
    cache['x_minus_mean'] = x_minus_mean
    cache['x_minus_mean_squared'] = x_minus_mean_squared
    cache['var'] = sample_var
    cache['sqrt_var'] = sqrt_var
    cache['i_sqrt_var'] = i_sqrt_var
    cache['x_hat'] = x_hat
    cache['eps'] = eps
    cache['gamma'] = gamma
    cache['beta'] = beta


    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_norm = (x - running_mean) / np.sqrt(running_var + eps)
    y = gamma * x_norm + beta
    out = y
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x = cache['x']
  mean = cache['mean']
  x_minus_mean = cache['x_minus_mean']
  x_minus_mean_squared = cache['x_minus_mean_squared']
  var = cache['var']
  sqrt_var = cache['sqrt_var']
  i_sqrt_var = cache['i_sqrt_var']
  x_hat = cache['x_hat']
  eps = cache['eps']
  gamma = cache['gamma']
  beta = cache['beta']

  N, D = dout.shape


  # ---------- 4 - scale and shift ----------

  #y = gamma * x_hat + beta
  dy = dout
  d_x_hat = dy * gamma #mult by dy - ChainRule

  # ---------- 3 - normalize ----------

  #x_hat = x_minus_mean * i_sqrt_var
  d_x_minus_mean = d_x_hat * i_sqrt_var #mult by d_x_hat - ChainRule
  d_i_sqrt_var = d_x_hat * x_minus_mean
  # sum up the gradients over dimension N, because the multiplication was row-wise during the forward pass.
  d_i_sqrt_var = np.sum(d_i_sqrt_var, axis=0)


  # x_minus_mean = x - sample_mean
  d_x = d_x_minus_mean * 1
  d_mean = np.sum(d_x_minus_mean, axis=0) * (-1)

  # i_sqrt_var = 1. / sqrt_var
  d_sqrt_var = d_i_sqrt_var * (-1.0 / sqrt_var**2)

  # sqrt_var = np.sqrt(sample_var + eps)
  #d_var = d_sqrt_var * (0.5 / np.sqrt(var + eps))
  d_var = d_sqrt_var * (0.5 / sqrt_var)

  # ---------- 2 - compute variance ----------

  #sample_var = 1. / N * np.sum(x_minus_mean_squared, axis=0)
  #column-wise summation during the forward pass,
  #during the backward pass means that we evenly distribute the gradient over all rows for each column.
  d_x_minus_mean_squared = d_var * (1. / N * np.ones((N, D)))

  # x_minus_mean_squared = x_minus_mean ** 2
  d_x_minus_mean = d_x_minus_mean_squared * (2 * x_minus_mean)

  # x_minus_mean = x - sample_mean
  d_x += d_x_minus_mean * 1
  d_mean += np.sum(d_x_minus_mean, axis=0) * (-1)

  # ---------- 1 - compute mean ----------
  #sample_mean = x.mean(axis=0)
  d_x += d_mean * (1. / N * np.ones((N, D)))

  dx = d_x

  # y = gamma * x_hat + beta
  dgamma = np.sum(dy * x_hat, axis=0)
  dbeta = np.sum(dy, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x = cache['x']
  mean = cache['mean']
  x_minus_mean = cache['x_minus_mean']
  x_minus_mean_squared = cache['x_minus_mean_squared']
  var = cache['var']
  sqrt_var = cache['sqrt_var']
  i_sqrt_var = cache['i_sqrt_var']
  x_hat = cache['x_hat']
  eps = cache['eps']
  gamma = cache['gamma']
  beta = cache['beta']

  N, D = dout.shape

  dy = dout

  # My Solution - Does not Work!
  #sum_x_minus_mean = np.sum(x - mean, axis=0)
  #dx_hat = ((N-1)*(N*(var+eps) - (x - mean)*sum_x_minus_mean) ) / (N**2 * np.sqrt(var + eps) * (var + eps)**2)
  #dx = dy * gamma * dx_hat
  # dgamma = np.sum(dy * x_hat, axis=0)
  # dbeta = np.sum(dy, axis=0)

  # cthorey's Soluton - Does not work!
  # h = x
  # mu = 1. / N * np.sum(h, axis=0)
  # var = 1. / N * np.sum((h - mu) ** 2, axis=0)
  # dx = (1. / N) * gamma * (var + eps) ** (-1. / 2.) * (N * dy - np.sum(dy, axis=0))
  # dbeta = np.sum(dy, axis=0)
  # dgamma = np.sum((h - mu) * (var + eps) ** (-1. / 2.) * dy, axis=0)
  dbeta = np.sum(dout, axis=0)
  #dgamma = np.sum((x - mean) * (var + eps) ** (-1. / 2.) * dy, axis=0)
  #dx = (1. / N) * gamma * (var + eps) ** (-1. / 2.) * (N * dy - np.sum(dy, axis=0) - (x - mean) * (var + eps) ** (-1.0) * np.sum(dy * (x - mean), axis=0))

  var_plus_eps = var+eps
  var_plus_eps_pow = (var_plus_eps) ** (-1. / 2.)

  dgamma = np.sum((x_minus_mean) * var_plus_eps_pow * dy, axis=0)
  dx = (1. / N) * gamma * var_plus_eps_pow * (N * dy - np.sum(dy, axis=0) - (x_minus_mean) * (var_plus_eps) ** (-1.0) * np.sum(dy * (x_minus_mean), axis=0))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None


  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    # mask = np.random.binomial([np.ones(x.shape)], 1 - p)[0] * (1.0 / (1 - p))

    #mask = (np.random.rand(*x.shape) < p) / p

    mask = (np.random.rand(*x.shape) >= p) / (1.0 - p)

    #mask = np.random.rand(*x.shape)
    #mask[mask < p] = 0
    #mask[mask >= p] = 1.0 / (1 - p)

    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width W.
  We convolve each input with F different filters, where each filter spans all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']

  x_padded = x
  if pad > 0:
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)


  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape

  out_h = 1 + (H + 2 * pad - HH) / stride
  out_w = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, out_h, out_w))

  for item in xrange(N):
    xx = x_padded[item]
    for depth in xrange(F):
      ww = w[depth]
      bb = b[depth]

      curr_y = 0
      for out_y in xrange(out_h):
        curr_x = 0
        for out_x in xrange(out_w):
          out[item, depth, out_x, out_y] = np.sum(xx[:, curr_x:curr_x+WW, curr_y:curr_y+HH] * ww) + bb
          #out[item, depth, out_y, out_x] = np.sum(xx[:, curr_y:curr_y+HH, curr_x:curr_x+WW] * ww) + bb

          curr_x += stride

        curr_y += stride


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache

  stride = conv_param['stride']
  pad = conv_param['pad']

  x_padded = x
  if pad > 0:
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape

  out_h = 1 + (H + 2 * pad - HH) / stride
  out_w = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x_padded)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  for item in xrange(N):
    xx = x_padded[item]
    for depth in xrange(F):
      ww = w[depth]

      curr_y = 0
      for out_y in xrange(out_h):
        curr_x = 0
        for out_x in xrange(out_w):
          dd = dout[item, depth, out_x, out_y]

          dx[item, :, curr_x:curr_x+WW, curr_y:curr_y+HH] += dd*ww
          dw[depth] += dd*xx[:, curr_x:curr_x+WW, curr_y:curr_y+HH]
          db[depth] += dd

          curr_x += stride

        curr_y += stride
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  if pad > 0:
    dx = dx[:,:,pad:dx.shape[2]-pad, pad:dx.shape[3]-pad]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  out_width = (W - pool_width) / stride + 1
  out_height = (H - pool_height) / stride + 1
  out = np.zeros((N, C, out_height, out_width))

  for i in xrange(N):
    for depth in xrange(C):
      in_y = 0
      for out_y in xrange(out_height):
        in_x = 0
        for out_x in xrange(out_width):
          out[i, depth, out_y, out_x] = np.max(x[i, depth, in_y:in_y+stride, in_x:in_x+stride])
          in_x += stride

        in_y += stride
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  out_width = (W - pool_width) / stride + 1
  out_height = (H - pool_height) / stride + 1

  dx = np.zeros_like(x)

  for i in xrange(N):
    for depth in xrange(C):
      in_y = 0
      for out_y in xrange(out_height):
        in_x = 0
        for out_x in xrange(out_width):
          local_ind = np.argmax(x[i, depth, in_y:in_y + stride, in_x:in_x + stride])
          dx_y = in_y + (local_ind / pool_width)
          dx_x = in_x + (local_ind % pool_width)
          dx[i, depth, dx_y, dx_x] += 1*dout[i, depth, out_y, out_x]
          in_x += stride

        in_y += stride
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  (N, C, H, W) = x.shape

  x = np.reshape(x, (N, C*H*W))

  gamma = np.repeat(gamma, H*W)
  gamma = np.reshape(gamma, C*H*W)

  beta = np.repeat(beta, H*W)
  beta = np.reshape(beta, C*H*W)

  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = np.reshape(out, (N, C, H, W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  """
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """

  (N, C, H, W) = dout.shape
  dout = np.reshape(dout, (N, C*H*W))
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)

  dx = np.reshape(dx, (N, C, H, W))

  dgamma = np.reshape(dgamma, (C, H, W))
  dgamma = np.sum(np.sum(dgamma, axis=1), axis=1)

  dbeta = np.reshape(dbeta, (C, H, W))
  dbeta = np.sum(np.sum(dbeta, axis=1), axis=1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
