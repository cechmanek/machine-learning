# frequently used deen Q learning network
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):
  def __init__(self, input_shape, num_actions):
    super().__init__()

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


class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super().__init__(in_features, out_features, bias)

    self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    
    if bias:
      self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
      self.register_buffer('epsilon_bias', torch.zeros(out_features))

    self.reset_parameters()

  def reset_parameters(self): # overloading initial method
    std = math.sqrt(3 / self.in_features)
    self.weight.data.uniform_(-std, std)
    self.bias.data.uniform_(-std, std)

  def forward(self, x):
    self.epsilon_weight.normal_()
    bias = self.bias
    if bias is not None:
      self.epsilon_bias.normal_()
      bias = bias + self.sigma_bias * self.epsilon_bias.data

   return F.linear(input, self.weight, + self.sigma_weight * self.epsilon_weight.data, bias) 