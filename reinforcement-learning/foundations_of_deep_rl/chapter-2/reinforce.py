from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99


class Pi(nn.Module):
  """ our REINFORCE policy """
  def __init__(self, in_dim, out_dim):
    super(Pi, self).__init__()

    layers = [nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)]
    self.model = nn.Sequential(*layers)
      
    self.onpolicy_reset()

    self.train() # set to training mode

  def onpolicy_reset(self):
    self.log_probs = []
    self.rewards = []

  def forward(self, x):
    return self.model(x)    

  def act(self, state):
    x = torch.from_numpy(state.astype(np.float32)) # convert to torch tensor
    pdparam = self.forward(x) # forward pass
    pd = Categorical(logits=pdparam) ## creates a categorical probability distribution
    action = pd.sample() # pi(a|s) in action via pd
    log_prob = pd.log_prob(action) # log_prob of pi(a|s)
    self.log_probs.append(log_prob) # store for training

    return action.item()


def train(pi, optimizer):
  # inner gradient ascent loop of REINFORCE algorithm
  T = len(pi.rewards)
  rets = np.empty(T, dtype=np.float32) # the returns
  future_ret = 0.0
  #compute the returns efficiently
  for t in reversed(range(T)): ## iterating over the episode backward since we know future reward of final_state==0, future reward of final_state-1 == final_state reward 
    future_ret = pi.rewards[t]  + gamma * future_ret
    rets[t] = future_ret

  rets = torch.tensor(rets) ## what's the difference between torch.tensor and torch.Tensor?
  log_probs = torch.stack(pi.log_probs)
  
  loss = -log_probs * rets # gradient term: Negative for maximizing
  loss = torch.sum(loss)

  optimizer.zero_grad()
  loss.backward() # backpropogate, compute gradients
  optimizer.step() # gradient ascent, update the weights

  return loss


def main():
  env = gym.make('CartPole-v0')
  in_dim = env.observation_space.shape[0] # 4 for for CartPole
  out_dim = env.action_space.n # 2 for CartPole

  pi = Pi(in_dim, out_dim) # policy network pi_theta for REINFORCE
  optimizer = optim.Adam(pi.parameters(), lr=0.01)

  for episode in range(350):
    state = env.reset()
    for t in range(200): # cartpole max timestep is 200
      action = pi.act(state)
      state, reward, done, _ = env.step(action)
      pi.rewards.append(reward)
      #env.render()
      
      if done:
        break

    loss = train(pi, optimizer)
    total_reward = sum(pi.rewards)
    solved = total_reward > 195.0 ## consider problem solved if we stay balanced for 195/200 timesteps

    pi.onpolicy_reset() # onpolicy: clear memory after training
    ## since REINFORCE in on policy algorithm we can only train once per episode
    ## we could do multiple optimizer steps, but we can't use a replay buffer
    
    print(f'Episode: {episode}, loss: {loss},\
          total_reward: {total_reward}, sovled: {solved}')


if __name__ == '__main__':
  main()
