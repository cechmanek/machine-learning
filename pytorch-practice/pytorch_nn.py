# example using torch.nn neural network module equivalent to warm_up_numpy.py example

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

# create our model. It holds all the necessary weights
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out)
                            )

# define our loss function. there are several built in
loss_fn = torch.nn.MSELoss(reduction='sum') # mean square error as in other examples

# train model via series of foward, backward, and weight update steps
learning_rate = 1e-6
for t in range(500):
  #forward pass
  y_pred = model(x) # call model object like a function

  # compute and print loss
  loss = loss_fn(y_pred, y)
  print('loss at step {} is {}'.format(t, loss.item()))

  # clear the gradient buffers before computing loss
  model.zero_grad()

  # now backprop to get the gradients. no need to hand compute gradients! it's all done for us!
  loss.backward()
  
  # update weights
  with torch.no_grad(): # wrap in no_grad() as we don't want to track this update step
    for param in model.parameters():
      param -= learning_rate * param.grad
