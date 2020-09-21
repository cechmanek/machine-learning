# example using torch.nn  and torch.optim modules equivalent to warm_up_numpy.py example

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

# create the optimizer
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train model via series of foward, backward, and weight update steps
for t in range(500):
  #forward pass
  y_pred = model(x) # call model object like a function

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