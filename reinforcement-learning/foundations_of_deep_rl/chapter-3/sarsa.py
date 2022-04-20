# implementation of SARSA algorithm
# from "Foundations of Deep Reinforcement Learning", see page 67

from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


GAMMA = 0.99
EPSILON_MIN = 0.01
BATCH_SIZE = 32


class SARSA(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(SARSA, self).__init__()

    layers = [nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)]
    self.model = nn.Sequential(*layers)

    self.epsilon = 1.0 # initial value
    self.epsilon_decay = 0.99 # rate to decay epsilon exploration factor
    self.num_actions = out_dim

    self.rewards = []
    self.experiences = []

    self.train() # set network to track gradients

  def forward(self, x):
    return self.model(x)

  def act(self, state):
    # epsilon-greedy policy
    random_num = torch.rand((1)).item()
    if random_num < self.epsilon:
     action = torch.randint(low=0, high=self.num_actions, size=(1,)).item()
    else:
      x = torch.from_numpy(state.astype(np.float32)) # convert to torch tensor
      q_values = self.forward(x) # forward pass

      best_action_value, best_action_index = torch.max(q_values, 0)
      action = best_action_index.item()
    return action

  def reset_buffers(self):
    self.rewards = []
    self.experiences = []

  def store_experience(self, experience):
     self.experiences.append(experience) 


def train(sarsa, optimizer):
    optimizer.zero_grad()
    # do batch SGD, where batch size is a global param
    states = torch.empty(len(sarsa.experiences), 4, dtype=torch.float32)
    actions = torch.empty(len(sarsa.experiences), 1, dtype=torch.int64)
    rewards = torch.empty(len(sarsa.experiences), 1, dtype=torch.float32)
    next_states = torch.empty(len(sarsa.experiences), 4, dtype=torch.float32)
    next_actions = torch.empty(len(sarsa.experiences), 1, dtype=torch.int64)
    dones = torch.empty(len(sarsa.experiences), 1, dtype=torch.float32)

    for index, exp in enumerate(sarsa.experiences):
      state, action, reward, next_state, next_action, done = exp
      states[index] = torch.tensor(state)
      actions[index] = torch.tensor(action)
      rewards[index] = torch.tensor(reward)
      next_states[index] = torch.tensor(next_state)
      next_actions[index] = torch.tensor(next_action)
      dones[index] = torch.tensor(done)

    predicted_q_values = sarsa(states)
    with torch.no_grad(): # no need to compute gradient at this point
      next_q_values = sarsa(next_states)

    predicted_q_values = predicted_q_values.gather(1, actions)
    next_q_values = next_q_values.gather(1, next_actions) # next q values of actions we took

    # determine the target Q values to train the network to: target_Q at time=t --> Q_t(s,a) = r_t + GAMMA * Q_t+1(s`,a`)
    #target_q_values = reward + GAMMA * sarsa(next_state)[next_action] * (not done) # aka labels for q network
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones) # equivalent to line above
    
    loss = nn.MSELoss()(target_q_values, predicted_q_values)
    loss.backward()
    optimizer.step()
  
    sarsa.epsilon = max(sarsa.epsilon * sarsa.epsilon_decay, EPSILON_MIN)
    return loss.item()


def main():
  env = gym.make('CartPole-v0')
  in_dim = env.observation_space.shape[0] # 4 for for CartPole
  out_dim = env.action_space.n # 2 for CartPole

  sarsa = SARSA(in_dim, out_dim) # Q Network for SARSA
  #optimizer = optim.Adam(sarsa.parameters(), lr=0.01)
  optimizer = optim.RMSprop(sarsa.parameters(), lr=0.01)

  all_rewards = []
  all_losses = []
  for episode in range(500):
    episode_rewards = []
    state = env.reset()
    for t in range(200): # cartpole max timestep is 200. Could do fewer steps than a full episode
      action = sarsa.act(state)
      next_state, reward, done, _ = env.step(action)
      episode_rewards.append(reward)
      next_action = sarsa.act(next_state) # decide what next action will be, but don't take it yet
      sarsa.experiences.append((state, action, reward, next_state, next_action, done))
      sarsa.rewards.append(reward)
      #env.render()

      state = next_state
      if done:
        break
        
      if len(sarsa.experiences) >= BATCH_SIZE:
        loss = train(sarsa, optimizer)
        sarsa.reset_buffers()
        all_losses.append(loss)

    total_reward = sum(episode_rewards)
    solved = total_reward > 195.0 # consider problem solved if we stay balanced for 195/200 timesteps

    print(f'Episode: {episode}, total_reward: {total_reward}, sovled: {solved}')

    all_rewards.append(total_reward)

  
if __name__ == "__main__":
  main()
