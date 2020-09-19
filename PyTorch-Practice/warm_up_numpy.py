# as a warmup we'll manually code a fully connected ReLU with one hidden layer and no bias in numpy.

import numpy as np

N = 64 # batch size
D_in = 1000 # input dimenion
H = 100 # hidden dimension
D_out = 10 # output dimension

# create random input and output/target data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# train model via series of foward, backward, and weight update steps
learning_rate = 1e-6
for t in range(500):
  #forward pass
  h = x.dot(w1)
  h_relu = np.maximum(h, 0)
  y_pred = h_relu.dot(w2)

  # compute and print loss
  loss = np.square(y_pred - y).sum()
  print('loss at step {} is {}'.format(t, loss))

  # back propogate w.r.t. loss to compute gradients of w1 and w2, then update weights
  grad_y_pred = 2.0 * (y_pred -y) # hand-computed derivative of y w.r.t loss == SUM(sqrt(y_pred-y))
  grad_w2 = h_relu.T.dot(grad_y_pred) # hand-computed derivative of w2 w.r.t. loss
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h<0] = 0
  grad_w1 = x.T.dot(grad_h)

  # update weights
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2