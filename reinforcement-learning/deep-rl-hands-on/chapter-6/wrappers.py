# gym environment wrapper around Atari pong

import cv2
import gym
import gym.spaces
import numpy as np
import collections


# to start many games you have to press the 'fire' button as the first action
# it's possible to learn this, but really a waste of time as it's not an interesting state or action
class FireResetEnv(gym.Wrapper):
  def __init__(self, env=None):
    super().__init__(env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unnwrapped.get_actino_meanings()) >= 3

  def step(self, action):
    return self.env.step(action)
  
  def reset(self):
    self.env.reset()
    obs, reward, is_done, _ = self.env.step(1) # press 'Fire' as first action always
    if is_done:
      self.env.reset()
    obs, reward, is_done, _ = self.env.step(2) # other corner case that some atari games have
    if is_done:
      self.env.reset()
    return obs

# for many games there's not point having a new action at every timestep,
# so skip a few frames between changing actions
class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env=None, skip=4):
    super.__init__(env)
    self.observation_buffer = collections.deque(maxlen=2) # observations are screen images
    self.skip = skip 

  def step(self, action):
    total_reward = 0.0
    is_done = False
    for _ in range(self.skip): # repeat the same action a given number of frames
      observation, reward, is_done, info = self.env.step(action)
      self.observation_buffer.append(observation)
      total_reward += reward
      if is_done:
       break 
    # to solve screen flickering take the max pixel values of our observations
    max_frame = np.max(np.stack(self.observation_buffer), axis=0)
    return (max_frame, total_reward, is_done, info)

  def reset(self):
    self.observation_buffer.clear()
    observation = self.env.reset()
    self.observation_buffer.append(obs)
    return observation
