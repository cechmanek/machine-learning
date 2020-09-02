# using state value iteration and the bellman optimality equation to solve frozen lake environment.
import gym
import collections
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TRAIN_EPISODES = 100
TEST_EPISODES = 20


class Agent():
  def __init__(self):
    self.env = gym.make(ENV_NAME)
    self.state = self.env.reset()

    self.rewards = collections.defaultdict(float) # regular dict with key:val -> (S0,A,S1):reward
    self.transitions = collections.defaultdict(collections.Counter) # dict of dicts
    self.state_values = collections.defaultdict(float) # key: val -> state_id: state_value

  def play_n_random_steps(self, count):
    # choose a random action and take it
    action = self.env.action_space.sample()

    new_state, reward, is_done, _ = self.env.step(action)
    # we only get reward on terminal state and it's always 1 so no need to sum then average rewards
    self.rewards[(self.state, action, new_state)] = reward 

    # track transition counts because they're stochastic. need to average and estimate later 
    self.transitions[(self.state, action)] [new_state] += 1  # key=(S0,A), val=dict(S1:count)

    if is_done:
      self.state = self.env.reset()
    else:
      self.state = new_state

  def calculate_action_value(self, state, action):
    action_value = 0.0

    # find probability of transitioning into state_n given this action
    total_transitions = sum(self.transitions[(state, action)].values())

    for new_state, count in self.transitions[(state, action)].items():
      reward = self.rewards[(state, action, new_state)]
      action_value += count/total_transitions * (reward + GAMMA*self.state_values[new_state])
      # value is probability of getting to new_state * (reward + GAMMA*new_state_value)
    return action_value
  
  def select_action(self, state):
    # choose the best action in the current state based on our learning so far
    # it's just the max of the available action values
    best_action = None
    best_value = None
    for action in range(self.env.action_space.n):
      action_value = self.calculate_action_value(state, action)
      if best_action is None or action_value > best_value:
        best_action = action
        best_value = action_value
    
    return best_action

  def play_episode(self, env):
    total_reward = 0.0
    state = env.reset() # use a new env as we don't know what state self.env is in
    while True:
      action = self.select_action(state)
      new_state, reward, is_done, _ = env.step(action)
      total_reward += reward

      # we can still learn while following our good policy, so we may as well
      self.rewards[(state, action, new_state)] = reward
      self.transitions[(state, action)][new_state] += 1

      if is_done:
        break
      state = new_state
    return total_reward

  def value_iteration(self):
    # now go through our self.state_values dict and update our estimates
    for state in range(self.env.observation_space.n):
      action_vals = [self.calculate_action_value(state, a) for a in range(self.env.action_space.n)]
      self.state_values[state] = max(action_vals)

## main section
if __name__ == "__main__":
  # run the agent on some random steps, then test it by playing an episode
  # record results on our SummaryWriter

  writer = SummaryWriter(comment='value-iteration')
  my_agent = Agent()
  test_env = gym.make('FrozenLake-v0')

  iteration = 0
  best_reward = 0.0
  while True:
    iteration += 1
    # train agent
    my_agent.play_n_random_steps(TRAIN_EPISODES)
    my_agent.value_iteration()

    # now test it
    reward = 0.0
    for _ in range(TEST_EPISODES):
      reward += my_agent.play_episode(test_env)
      writer.add_scalar('reward', reward, iteration)
      if reward > best_reward:
        print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
        best_reward = reward
    if reward > 0.80:
      print("Solved in %d iterations!" % iteration)
      break
  writer.close()