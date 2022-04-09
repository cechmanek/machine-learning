# code snippet showing how to implement a policy network to handle continuous action spaces
# see page 38 of 'Foundations of Deep Reinforcement Learning'

from torch.distributions import Normal
import torch

# suppose for 1 action (Pendulum environment, action = torque applied)
# we obtain its mean and std dev from a policy network
policy_net_output = torch.tensor([1.0, 0.2])
# the pdparams are (mean, st_ dev), also commonly called (location, scale)
pdparams = policy_net_output
pd = Normal(loc=pdparams[0], scale=pdparams[1])

# sample an action
action = pd.sample()
# => tensor(1.0295), the amount of torque to apply

# compute action log probability
pd.log_prob(action)
# => tensor(0.6796), log probability of this torque
