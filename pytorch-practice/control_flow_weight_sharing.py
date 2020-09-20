# example demonstrating the usefulness of dynamic graphs
# build a fully connected network that on each pass randomly chooses between 1 and 4 hidden layers

import random
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
class DynamicNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super().__init__()
    # define all layers in __init__ so they get initialized only once
    self.input_linear = torch.nn.Linear(D_in, H)
    self.middle_linear = torch.nn.Linear(H, H)
    self.output_linear = torch.nn.Linear(H, D_out)

  def forward(self, x):
    x = self.input_linear(x)
    # repeat the same layer 1,2, or 3 times
    for _ in range(random.randint(0,3)):
      x = torch.nn.ReLU()(self.middle_linear(x))
    x = self.output_linear(x)
    return x

# instantiate our model
my_model = DynamicNet(D_in, H, D_out)

# define our loss function. there are several built in
loss_fn = torch.nn.MSELoss(reduction='sum') # mean square error as in other examples

# create the optimizer
learning_rate = 1e-4 # higher learning rate than other examples, just because
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
# train model via series of foward, backward, and weight update steps
for t in range(500):
  #forward pass
  y_pred = my_model(x) # call model object like a function

  # compute and print loss
  loss = loss_fn(y_pred, y)
  print('loss at step {} is {}'.format(t, loss.item()))

  # still need to clear gradient buffers manually
  # can do this vis model.zero_grad() or optmizer.zero_grad()
  optimizer.zero_grad()

  # compute gradients via back prop
  loss.backward()

  # take one step
  optimizer.step()