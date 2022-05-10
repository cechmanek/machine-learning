# WIP. currently has the same code as dqn_target_network.py
# implementation of double DQN algorithm with target network, uses Boltzmann exploration,
# not epsilon-greedy.
# from "Foundations of Deep Reinforcement Learning", see page 109
# for comparison to standard DQN see ../chapter-4/dqn.py
from torch.utils.data import WeightedRandomSampler
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


GAMMA = 0.99
TAU_MIN = 0.1
BATCH_SIZE = 8
REPLAY_BUFFER_SIZE = 10000
TARGET_NET_UPDATE_RATE = 25


class DQN(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(DQN, self).__init__()

    layers = [nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)]
    self.model = nn.Sequential(*layers)

    self.tau = 5.0 # initial value
    self.tau_decay = 0.99 # rate to decay tau anealling factor
    self.num_actions = out_dim

    self.rewards = []
    self.experiences = deque(maxlen=REPLAY_BUFFER_SIZE)

    self.train() # set network to track gradients

  def forward(self, x):
    return self.model(x)

  def act(self, state):
    # Boltzmann policy
    # this is a soft-max on the raw Q values, with a tau anealling factor that promotes exploration
    x = torch.from_numpy(state.astype(np.float32)) # convert to torch tensor
    q_values = self.forward(x) # forward pass
    boltzmann = nn.Softmax()
    boltzmann_action_probabilities = boltzmann(q_values/self.tau)

    # now take a weighted random sample from these probabilities to determine which action to take
    action = WeightedRandomSampler(boltzmann_action_probabilities, 1)
    return list(action)[0]

    '''
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
    '''

  def reset_buffers(self):
    self.rewards = []
    
  def store_experience(self, experience):
    self.experiences.append(experience) 
    # no need to drop all experiences we've already trained on
    # just be sure that we we drop oldest experiences from buffer as we reach buffer size limit
    #self.experiences = self.experiences[len(self.experiences)- REPLAY_BUFFER_SIZE:


def train(dqn, target_dqn, optimizer):
    optimizer.zero_grad()
    # do batch SGD, where batch size is a global param
    states = torch.empty(BATCH_SIZE, 4, dtype=torch.float32)
    actions = torch.empty(BATCH_SIZE, 1, dtype=torch.int64)
    rewards = torch.empty(BATCH_SIZE, 1, dtype=torch.float32)
    next_states = torch.empty(BATCH_SIZE, 4, dtype=torch.float32)
    dones = torch.empty(BATCH_SIZE, 1, dtype=torch.float32)

    #randomly choose which prior experiences to train on
    indices = np.random.choice(len(dqn.experiences), BATCH_SIZE, replace=False)
    for i, index in enumerate(indices):
      ##state, action, reward, next_state, done = exp
      state, action, reward, next_state, done = dqn.experiences[index]
      states[i] = torch.tensor(state)
      actions[i] = torch.tensor(action)
      rewards[i] = torch.tensor(reward)
      next_states[i] = torch.tensor(next_state)
      dones[i] = torch.tensor(done)

    predicted_q_values = dqn(states)
    with torch.no_grad(): # no need to compute gradient at this point
      next_q_values = target_dqn(next_states) # use target net to predict target q values, for stability

    predicted_q_values = predicted_q_values.gather(1, actions)
    next_q_values = next_q_values.max(1)[0].unsqueeze(-1) # best q values of next state
    # for double DQN the above line should use actions chosen by dqn(states), not max of target_dqn(states)
    # next_q_values = next_q_values.[actions].unsqueeze(-1) # something like this for double DQN

    # determine the target Q values to train the network to: target_Q at time=t --> Q_t(s,a) = r_t + GAMMA * max(Q_t+1(s`))
    #target_q_values = reward + GAMMA * max(target_dqn(next_state)) * (not done) # aka labels for q network
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones) # equivalent to line above
    
    loss = nn.MSELoss()(target_q_values, predicted_q_values)
    loss.backward()
    optimizer.step()
  
    dqn.tau = max(dqn.tau * dqn.tau_decay, TAU_MIN)
    return loss.item()


def main():
  env = gym.make('CartPole-v0')
  in_dim = env.observation_space.shape[0] # 4 for for CartPole
  out_dim = env.action_space.n # 2 for CartPole

  dqn = DQN(in_dim, out_dim) # Q Network for DQN
  target_dqn = DQN(in_dim, out_dim) # target Q Network for DQN to add stability to training
  #optimizer = optim.Adam(dqn.parameters(), lr=0.01)
  optimizer = optim.RMSprop(dqn.parameters(), lr=0.01)

  all_rewards = []
  all_losses = []
  for episode in range(500):
    episode_rewards = []
    state = env.reset()
    for t in range(200): # cartpole max timestep is 200. Could do fewer steps than a full episode
      action = dqn.act(state)
      next_state, reward, done, _ = env.step(action)
      episode_rewards.append(reward)
      ##next_action = dqn.act(next_state) # decide what next action will be, but don't take it yet
      ##dqn.experiences.append((state, action, reward, next_state, next_action, done))
      dqn.experiences.append((state, action, reward, next_state, done))
      dqn.rewards.append(reward)
      #env.render()

      state = next_state
      if done:
        break
        
      if len(dqn.experiences) >= BATCH_SIZE and t % 5 == 0: # no need to train every single step 
        loss = train(dqn, target_dqn, optimizer)
        dqn.reset_buffers()
        all_losses.append(loss)

    if episode & TARGET_NET_UPDATE_RATE == 0:
        #set target_dqn.parameters = dqn.parameters # target_dqn won't change every train step, so is more stable
        target_dqn.load_state_dict(dqn.state_dict(), strict=True)

    total_reward = sum(episode_rewards)
    solved = total_reward > 195.0 # consider problem solved if we stay balanced for 195/200 timesteps

    print(f'Episode: {episode}, total_reward: {total_reward}, sovled: {solved}')

    all_rewards.append(total_reward)

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(all_rewards)
  plt.title('rewards')
    
  plt.figure()
  plt.plot(all_losses)
  plt.title('loss')

if __name__ == "__main__":
  main()
