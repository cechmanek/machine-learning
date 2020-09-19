# using the cross-entropy method to learn to navigate a slippery frozen lake

import gym
import random
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

HIDDEN_SIZE = 128 # how many neurons in our single hidden layer neural net
BATCH_SIZE = 100
PERCENTILE = 30 # what threshold to cut off 'good' versus 'bad' examples
GAMMA = 0.9 # discount factor for future rewards

# frozen lake env observation is just a number representing agent's position in a gridworld.
# we need to one hot encode this so that we have a consistent input vector for our neural net.
class DiscreteOneHotWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

  def observation(self, observation):
    res = np.copy(self.observation_space.low)
    res[observation] = 1.0
    return res

class Net(nn.Module):
  def __init__(self, obs_size, hidden_size, n_actions):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, n_actions) # no softmax output. it'll be applied later
    )

  def forward(self, x):
    return self.net(x)


# some declarations and helper functions
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation','action'])

def iterate_batches(env, net, batch_size):
  batch = []
  episode_reward = 0.0 
  episode_steps = []
  obs = env.reset()
  while True: # continue in this episode until we finish
    obs_vector = torch.tensor([obs], dtype=torch.float) # torch needs a batch of examples
    action_probs_vector = nn.Softmax(dim=1)(net(obs_vector)) # action probabilities
    action_probs = action_probs_vector.data.numpy()[0] # pull first (and only) example out

    # choose an action randomly, but weighted by probabilities
    action = np.random.choice(len(action_probs), p=action_probs)
    next_obs, reward, is_done, _ = env.step(action)
    episode_reward += reward
    episode_steps.append(EpisodeStep(observation=obs, action=action))

    if is_done: # finished one episode, so append it and get ready for next
      batch.append(Episode(reward=episode_reward, steps=episode_steps))
      # reset things for next of episode
      episode_reward = 0.0
      episode_steps = []
      next_obs = env.reset()
      # return batch once it's full
      if len(batch) == batch_size:
        yield batch
        batch = []
    obs = next_obs

def filter_batch(batch, percentile):
  discounted_rewards = [s.reward*(GAMMA**len(s.steps)) for s in batch]
  reward_bound = np.percentile(discounted_rewards, percentile)

  train_obs = []
  train_actions = []
  elite_batch = [] # we don't see many good examples so keep the ones we do
  for example, reward in zip(batch, discounted_rewards):
    if reward < reward_bound:
      continue # we did poorly on this example, don't train on it
    train_obs.extend([step.observation for step in example.steps])
    train_actions.extend([step.action for step in example.steps])
    elite_batch.append(example)

    train_obs_vector = torch.tensor(train_obs, dtype=torch.float)
    train_actions_vector = torch.tensor(train_actions, dtype=torch.long)

  return elite_batch, train_obs_vector, train_actions_vector, reward_bound

# main process
if __name__ == "__main__":
  random.seed(12345)
  env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0'))
  env = gym.wrappers.Monitor(env, directory='monitor', force=True) # records video

  obs_size = env.observation_space.shape[0]
  num_actions = env.action_space.n

  # set up neural net and optimizer
  net = Net(obs_size, HIDDEN_SIZE, num_actions)
  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=net.parameters(), lr=0.001)
  writer = SummaryWriter(comment='-frozenlake-tweaked')

  # now generate batches and train on them
  full_batch = []
  for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
    reward_mean = np.mean([s.reward for s in batch])
    # keep passing full_batch so that full_batch will always contain all our good examples
    full_batch, obs_vector, actions_vector, reward_bound = filter_batch(full_batch + batch, PERCENTILE)

    if len(full_batch) == 0:
      continue # if we have 0 good examples to train on. go get some more

    full_batch = full_batch[-500:] # keep up to 500 good examples. this is safe if len(full_b) < 500

    optimizer.zero_grad()
    action_scores_vector = net(obs_vector)
    loss = objective(action_scores_vector, actions_vector)
    loss.backward()
    optimizer.step()
    
    print("iteraion:%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
           %(iter_no, loss.item(), reward_mean, reward_bound))
    
    writer.add_scalar("loss", loss.item(), iter_no)
    writer.add_scalar("reward_bound", reward_bound, iter_no)
    writer.add_scalar("reward_mean", reward_mean, iter_no)

    if reward_mean > 0.8:
      print('we solved FrozenLake')
      break #iterate_batches continues indefinitely so break once our agent is basically perfect

  writer.close()