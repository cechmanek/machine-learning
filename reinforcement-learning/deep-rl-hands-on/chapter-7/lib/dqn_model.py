# frequently used deen Q learning network
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):
  def __init__(self, input_shape, num_actions):
    super.__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2D(64, 64, kernel_size=3, stride=1),
      nn.ReLU()
    )

    conv_out_size = self.get_conv_out_size(input_shape)

    self.dense = nn.Sequential(
      nn.Linear(conv_out_size, 512),
      nn.ReLU(),
      nn.Linear(512, num_actions)
    )

def get_conv_out_size(self, input_shape):
  dummy_input = torch.zeros(1, *input_shape)
  out = self.conv(dummy_input)
  return int(np.prod(out.size()))

def forward(self, x):
  fx = x.float() / 256
  conv_out = self.conv(fx).view(fx.size()[0],-1) # flatten output of conv layers
  return self.dense(conv_out)