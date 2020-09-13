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


