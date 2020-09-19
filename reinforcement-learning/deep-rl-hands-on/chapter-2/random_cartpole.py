
import gym
from cartpole_action_wrapper import RandomActionWrapper

if __name__ == "__main__":
  env = gym.make('CartPole-v0')   
  env = gym.wrappers.Monitor(env, 'output', force=True)

  observation = env.reset() # must always call after we initilizse a gym environment
  
  # set up agent params
  total_reward = 0.0
  total_steps = 0

  while True:
    action = env.action_space.sample() # random action selected
    observation, reward, done, details = env.step(action)

    # ignore observations for now
    total_reward += reward
    total_steps += 1
    if done:
      break

  print('Episode done in %d steps, total reward of %.2f' %(total_steps, total_reward))