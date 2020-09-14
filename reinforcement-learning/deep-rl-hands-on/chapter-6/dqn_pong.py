# training our deep Q-learning network to play Atari pong
# this file holds 2 identical deep networks, and periodically syncs their weights
# this helps to stabilize training
import wrappers
import dqn_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import collections

import numpy as np

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5 # average score over last 100 games to consider game solved

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
  def __init__(self, env, experience_buffer):
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
  next_state_values = target_net(next_states_tensor).max(1)[0] # .max() returns max and argmax
  next_state_values[done_mask] = 0.0 # if at terminal state we need to manually set next_state_val=0
  next_state_values = next_state_values.detach() # detach so we don't back prop through bellman eq

  expected_state_action_values = rewards_tensor + next_state_values * GAMMA

  return nn.MSELoss()(state_action_values, expected_state_action_values)


# now train our agent
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
  parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                      help="Name of the environment, default=" + DEFAULT_ENV_NAME)
  parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                      help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
  args = parser.parse_args()
  device = torch.device("cuda" if args.cuda else "cpu")

  # launch our pong environment with all the custom wrappers we defined
  env = wrappers.make_env(args.env)

  # create two networks
  # one that we use for getting subsequent state values, but doesn't change every iteration
  # this helps stabilize training
  net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
  target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

  writer = SummaryWriter(comment="-" + args.env)

  print('our deep Q network looks like:')
  print(net)

  # initialize replay buffer and agent
  buffer = ExperienceBuffer(REPLAY_SIZE)
  agent = Agent(env, buffer)
  epsilon = EPSILON_START

  # initialize our optimizer
  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
  total_rewards = []
  frame_idx = 0
  ts_frame = 0 # track time on a given frame
  ts = time.time() # track current time
  best_mean_reward = None

  # master loop for training
  while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
      total_rewards.append(reward)
      speed = (frame_idx - ts_frame) / (time.time() - ts)
      ts_frame = frame_idx
      ts = time.time()
      mean_reward = np.mean(total_rewards[-100:]) # average last 100 plays
      print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
          frame_idx, len(total_rewards), mean_reward, epsilon, speed))

      writer.add_scalar("epsilon", epsilon, frame_idx)
      writer.add_scalar("speed", speed, frame_idx)
      writer.add_scalar("reward_100", mean_reward, frame_idx)
      writer.add_scalar("reward", reward, frame_idx)

      if best_mean_reward is None or best_mean_reward < mean_reward:
        torch.save(net.state_dict(), args.env + "-best.dat") # save current model to sync later
        if best_mean_reward is not None:
          print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
        best_mean_reward = mean_reward
    
      if mean_reward > args.reward:
        print("Solved in %d frames!" % frame_idx)
        break

    if len(buffer) < REPLAY_START_SIZE:
      continue # don't train network until our replay buffer is full

    if frame_idx % SYNC_TARGET_FRAMES == 0:
      target_net.load_state_dict(net.state_dict()) # sync our target model with saved model

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, target_net, device=device)
    loss_t.backward()
    optimizer.step()

  writer.close()