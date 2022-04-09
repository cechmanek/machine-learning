# code snippet showing how to implement a policy network to handle discrete action spaces
# see page 37 of 'Foundations of Deep Reinforcement Learning'

from torch.distributions import Categorical
import torch


# suppose 2 actions (CartPole: move left, move right)
# we obtain their logit probabilities from a policy network
policy_net_output = torch.tensor([-1.6094, -0.2231])

# the pdparams are logits, equivalent to probabilities [0.2, 0.8]
pdparams = policy_net_output
pd = Categorical(logits=pdparams)

# sample an action
action = pd.sample()
# => tensor(1), or 'move right'

# compute the action log probability
pd.log_prob(action)
# => tensor(-0.2231), log probability of 'move right'
