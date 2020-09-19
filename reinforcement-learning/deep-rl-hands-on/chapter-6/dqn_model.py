# 5 layer convolutional net to process images and return all Q values of actions in a single pass
import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
  def __init__(self, input_shape, n_actions):
    super().__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU()
    )

    conv_out_size = self.get_conv_out(input_shape)
    self.fc = nn.Sequential(
      nn.Linear(conv_out_size, 512),
      nn.ReLU(),
      nn.Linear(512, n_actions)
    )

  def forward(self, x):
    conv_out = self.conv(x).view(x.size()[0], -1) # flatten output of conv layers
    return self.fc(conv_out) # return Q values of all actions in one pass
    
  def get_conv_out(self, shape): # get the size of the flattened output
    out = self.conv(torch.zeros(1, *shape)) # some dummy input just to get output of correct size
    return int(np.prod(out.size()))

 