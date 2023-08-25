
import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
  def __init__(self, env, epsilon):
    super().__init__(env)
    self.epsilon = epsilon

  def action(self, action):
    if random.random() < self.epsilon: # choose a random action instead
      print('chose a random action instead')
      return self.env.action_space.sample()
    return action


if __name__ == "__main__":

  env = RandomActionWrapper(gym.make('CartPole-v0'), 0.4)
  env = gym.wrappers.Monitor(env, 'output', force=True)

  observation = env.reset() # must always call after we initilizse a gym environment
  
  # set up agent params
  total_reward = 0.0
  total_steps = 0

  while True:
    action = 0 # always go left
    observation, reward, done, details = env.step(action)

    # ignore observations for now
    total_reward += reward
    total_steps += 1
    if done:
      break

  print('Episode done in %d steps, total reward of %.2f' %(total_steps, total_reward))