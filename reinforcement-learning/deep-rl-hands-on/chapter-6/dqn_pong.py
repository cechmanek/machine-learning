# training our deep Q-learning network to play Atari pong
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import collections

import numpy as np

DEFAULT_ENV_NAME = "PongNoFrameSkip-v4"
MEAN_REWARD_BOUNT = 19.5 # average score over last 100 games to consider game solved

GAMMA = 0.99 # discount factor in Bellman equations
BATCH_SIZE = 32 # number of observations to train on in one batch
REPLAY_SIZE = 10000 # size of our replay buffer that we draw batches from
REPLAY_START_SIZE = 10000 # wait until buffer has this many samples before we start training

SYNC_TARGET_FRAMES = 1000 # how often to transfer weights between training model and target model
LEARNING_RATE = 1e-4 # learning rate for deep neural network optimizer

EPSILON_DECAY_LAST_FRAME = 10**5 # decay epsilon from e_start to e_final over 100000 frame
EPSILON_START = 1.0 # epsilon is probability of choosing random actions. Used for early exploring
EPSILON_FINAL = 0.02 # as we learn more reduce the probability of random actions to this value


Experience = collections.namedtuple('Experience',
                                    field_names=['state','action','reward','done', 'new_state'])


class ExperienceBuffer():
  def __init__(self, capacity):
    self.buffer = collections.deque(maxlen=capacity)

  def __len__(self):
    return len(self.buffer)
  
  def append(self, experience):
    self.buffer.append(experience)


  def sample(self, batch_size):
    # randomly sample some experiences. This ensures we see things out of order while training
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
  

class Agent:
  def __init__(self, env, exp_buffer):
    self.env = env
    self.experience_buffer = experience_buffer
    self._reset()

  def _reset(self):
    self.state = env.reset()
    self.total_reward = 0.0

  def play_step(self, net, epsilon=0.0, device="cpu"):
    done_reward = None

    if np.random.random() < epsilon:
      action = env.action_space.sample()
    else:
      state_array = np.array([self.state], copy=False)
      state_tensor = torch.tensor(state_array).to(device)
      q_values = net(state_tensor) # predict values of all actions
      _, action_tensor = torch.max(q_values, dim=1) # get index of the action with maximum value
      action = int(action_tensor.item())

    # do step in the environment
    new_state, reward, is_done, _ = self.env.step(action)
    self.total_reward += reward

    #add this experience to our replay buffer
    exp = Experience(self.state, action, reward, is_done, new_state)
    self.experience_buffer.append(exp)
    self.state = new_state
    if is_done:
      done_reward = self.total_reward
      self._reset()
    return done_reward


def calculate_loss(batch, net, target_net, device='cpu'):
  states, actions, rewards, dones, next_states = batch

  states_tensor = torch.tensor(states).to(device)
  next_states_tensor = torch.tensor(next_states).to(device)
  actions_tensor = torch.tensor(actions).to(device)
  rewards_tensor = torch.tensor(rewards).to(device)
  done_mask = torch.ByteTensor(dones).to(device)

  state_action_values = net(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
  next_state_values = target_net(next_states_tensor).max(1)[0]
  next_state_values[done_mask] = 0.0
  next_state_values = next_state_values.detach()

  expected_state_action_values = rewards_tensor + next_state_values * GAMMA

  return nn.MSELoss()(state_action_values, expected_state_action_values)