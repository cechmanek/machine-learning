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


class ProcessFrame84(gym.Wrapper):
  def __init__(self, env=None):
    super__init__(env)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)

  def observation(self, obs):
    return ProcessFrame84.process(obs)

  @staticmethod
  def process(frame):
    if frame.size == 210 * 160 * 3:
      img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
      img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    else:
      raise EnvironmentError('unkown resolution')
    # convert to grayscale, but weighted to better match human perception
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * .0114

    resized_screen = cv2.resize(img, (84,110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :] # crop top & bottom of screen as it's useless. This saves memory
    x_t = np.reshape(x_t, [84, 81, 1]) # reshape to 84 by 84 pixels
    return x_t.astype(np.uint8)

