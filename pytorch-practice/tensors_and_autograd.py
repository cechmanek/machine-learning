# example using torch tensors & autograd in a neural network equivalent to warm_up_numpy.py example

import torch

dtype = torch.float
device = torch.device('cpu') # use 'cuda' or 'cuda:0' if we have multiple GPUs

N = 64 # batch size
D_in = 1000 # input dimension
H = 100 # hidden layer size
D_out = 10 # output dimension

# create random input and targets
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# train model via series of foward, backward, and weight update steps
learning_rate = 1e-6
for t in range(500):
  #forward pass
  h = x.mm(w1) # matrix_multiply. equvivalent to dot product
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print('loss at step {} is {}'.format(t, loss.item()))

  # back propogate w.r.t. loss to compute gradients of w1 and w2, then update weights
  loss.backward()
  
  # no need to hand compute gradients! it's all done for us!
  ''' 
  grad_y_pred = 2.0 * (y_pred -y) # hand-computed derivative of y w.r.t loss == SUM(sqrt(y_pred-y))
  grad_w2 = h_relu.t().mm(grad_y_pred) # hand-computed derivative of w2 w.r.t. loss
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h<0] = 0
  grad_w1 = x.t().mm(grad_h)

  ''' 
  # update weights
  with torch.no_grad(): # wrap in no_grad() as we don't want to track this update step
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    # after updating weights we need to clear the gradient buffer
    w1.grad.zero_()
    w2.grad.zero_()