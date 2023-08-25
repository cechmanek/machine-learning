# using tabular q-learing on frozenlake environment.
# only learn q values for states we encounter, not all possible states

import gymnasium as gym
import collections
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9 # discount factor
ALPHA = 0.2 # learning update rate for q values
TEST_EPISODES = 20

class Agent():
  def __init__(self):
    self.env = gym.make(ENV_NAME)
    self.state = self.env.reset()
    self.action_values = collections.defaultdict(float)

  def sample_env(self):
    action = self.env.action_space.sample()
    old_state = self.state
    new_state, reward, is_done, _ = self.env.step(action)

    if is_done:
      self.state = self.env.reset()
    else:
      self.state = new_state

    return (old_state, action, reward, new_state)

  def best_value_and_action(self, state):
    # loop through all possible actions in this state and find highest estimated value
    best_value = None
    best_action = None
    for action in range(self.env.action_space.n):
      action_value = self.action_values[(state, action)]
      if best_value is None or best_value < action_value:
        best_action = action
        best_value = action_value
    return (best_value, best_action)

  def my_value_update(self, state, action, reward, next_state):
    # update state-action pairs with learning update rate ALPHA and discount factor GAMMA
    current_value = self.action_values[(state, action)] 
    best_value_at_new_state, best_action_at_new_state = self.best_value_and_action(next_state) 
    new_value = (1-ALPHA)*current_value + ALPHA*(reward +GAMMA*best_value_at_new_state)
    self.action_values[(state, action)] = new_value

  def value_update(self, s, a, r, next_s):
    best_v, _ = self.best_value_and_action(next_s)
    new_val = r + GAMMA * best_v
    old_val = self.action_values[(s, a)]
    self.action_values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA


  def play_episode(self, env):
    # play one episode following our learned values
    # play on a new environment so we don't change the state of self.env
    total_reward = 0.0
    state =env.reset()
    while True:
      best_value, best_action = self.best_value_and_action(state)
      new_state, reward, is_done, _ = env.step(best_action)
      state = new_state
      total_reward += reward
      if is_done:
        break
      
    return total_reward


# now train and play
if __name__ == '__main__':
  agent = Agent()
  test_env = gym.make(ENV_NAME)
  writer = SummaryWriter(comment='q-learning_FrozenLake')

  iteration = 0
  best_reward  = 0.0
  while True:
    iteration += 1
    state, action, reward, next_state = agent.sample_env()
    agent.value_update(state, action, reward, next_state) 

    reward = 0.0
    for _ in range(TEST_EPISODES):
      reward += agent.play_episode(test_env)
    
    reward /= TEST_EPISODES
    writer.add_scalar('reward', reward, iteration)

    if reward > best_reward:
      best_reward = reward
      print('best reward so far {} on iteration {}'.format(best_reward, iteration))

    if reward > 0.8:
      print('solved {} in {} iterations'.format(ENV_NAME, iteration))
      break

  writer.close()