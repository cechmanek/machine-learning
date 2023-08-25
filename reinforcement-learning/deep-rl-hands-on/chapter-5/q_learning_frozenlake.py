# using action value iteration and the bellman optimality equation to solve frozen lake environment.
import gymnasium as gym
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
    self.action_values = collections.defaultdict(float) # key: val -> (state, action): action_value

  def play_n_random_steps(self, count):
    for _ in range(count):
      # choose a random action and take it
      action = self.env.action_space.sample()

      new_state, reward, terminated, truncated, _ = self.env.step(action)
      # we only get reward on terminal state and it's always 1 so no need to sum then average rewards
      self.rewards[(self.state, action, new_state)] = reward 

      # track transition counts because they're stochastic. need to average and estimate later 
      self.transitions[(self.state, action)] [new_state] += 1  # key=(S0,A), val=dict(S1:count)

      if terminated or truncated:
        self.state = self.env.reset()
      else:
        self.state = new_state

  def select_action(self, state):
    # choose the best action in the current state based on our learning so far
    # it's just the max of the available action values
    best_action = None
    best_value = None
    for action in range(self.env.action_space.n):
      action_value = self.action_values[(state, action)]
      if best_action is None or action_value > best_value:
        best_action = action
        best_value = action_value
    
    return best_action

  def play_episode(self, env):
    total_reward = 0.0
    state = env.reset() # use a new env as we don't know what state self.env is in
    while True:
      action = self.select_action(state)
      new_state, reward, terminated, truncated, _ = env.step(action)
      total_reward += reward

      # we can still learn while following our good policy, so we may as well
      self.rewards[(state, action, new_state)] = reward
      self.transitions[(state, action)][new_state] += 1

      if terminated or truncated:
        break
      state = new_state
    return total_reward

  def q_learning(self):
    # now go through our self.action_values dict and update our estimates
    for state in range(self.env.observation_space.n):
      for action in range(self.env.action_space.n):
        action_value = 0.0
        total_transitions = sum(self.transitions[(state, action)].values())
        for new_state, count in self.transitions[(state, action)].items():
          reward = self.rewards[(state, action, new_state)]
          best_action = self.select_action(new_state)
          action_value += (count/total_transitions) * (reward + GAMMA*self.action_values[(new_state, best_action)])
        
        self.action_values[(state, action)] = action_value

## main section
if __name__ == "__main__":
  # run the agent on some random steps, then test it by playing an episode
  # record results on our SummaryWriter

  writer = SummaryWriter(comment='q-learning-iteration')
  my_agent = Agent()
  test_env = gym.make('FrozenLake-v0')

  iteration = 0
  best_reward = 0.0
  while True:
    iteration += 1
    # train agent
    my_agent.play_n_random_steps(TRAIN_EPISODES)
    my_agent.q_learning()

    # now test it
    reward = 0.0
    for _ in range(TEST_EPISODES):
      reward += my_agent.play_episode(test_env)
    reward /= TEST_EPISODES

    writer.add_scalar('reward', reward, iteration)

    if reward > best_reward:
      print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
      best_reward = reward

    if reward > 0.80:
      print("Solved in %d iterations!" % iteration)
      break
  writer.close()