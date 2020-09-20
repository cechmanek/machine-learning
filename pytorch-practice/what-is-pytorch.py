import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__() # equivalent to super().__init__()

    # 1 input image channel, 6 output channels, 3x3 square convolution
    self.conv1 = nn.Conv2d(1,6,3)
    self.conv2 = nn.Conv2d(6,16,3)

    # add fully connected layers
    self.fc1 = nn.Linear(16*6*6, 120) # 6*6 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # max pool over 2x2 window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

    x = F.max_pool2d(F.relu(self.conv2(x)), 2) # equivalent to above
    x = x.view(-1, self.num_flat_features(x)) # flatten for fully connected layers

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x) # linear output
    return x
    
  def num_flat_features(self,x):
    size = x.size()[1:] # all dimensions except batch dim
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


net = Net()
print(net) 

# see the learnable parameters
print(net.parameters())
# look at conv1, aka first layer, weights
#print(list(net.parameters())[0])
print(list(net.parameters())[0].size())

# do a forward pass by calling net(), use some dummy data here
input_image = torch.randn(1,1,32,32) # batch size of 1
output = net(input_image) # this also populates data through the autograd functionality
print(output)

### loss function
target = torch.randn(10) # some random y labels
target = target.view(1, -1)
criterion = nn.MSELoss()


loss = criterion(output, target)
print(loss)

# follow the loss gradient function back to input_image
# the chain looks like:

# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#      -> view -> linear -> relu -> linear -> relu -> linear
#      -> MSELoss
#      -> loss

# print a few of these
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# now do back propagation starting from loss
net.zero_grad() # clear gradient buffers before each back prop step
print('conv1 grads before back prop', net.conv1.bias.grad)
loss.backward()
print('conv1 grads after back prop', net.conv1.bias.grad)

# Update the weights, aka training
# here we want to use a built in optimizer, that than to SGD ourselves
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

# our training loop would be:

all_data = torch.randn((10,1,1,32,32)) # 10 batches, of 1 gray image each

i=0
for epoch in range(5):
  print('on epoch:', epoch)
  i=0
  for batch in all_data:
    print('on batch: ', i)
    i+=1
    optimizer.zero_grad() # calls net.zero_grad()
    output = net(batch)
    loss = criterion(output, target) # should have different targets per batch
    loss.backward()
    optimizer.step()