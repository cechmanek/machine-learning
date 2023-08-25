# modified dqn model using PTAN library, now implementing n-step Bellman update
import gymnasium as gym
import ptan
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from lib import dqn_model, common # our local 'lib' folder

UNROLL_STEPS = 2

if __name__ == "__main__":
  params = common.HYPERPARAMS['pong'] # loads params from file rather than specifying them all here
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', default=False, action='store_true', help='Enables cuda computation')
  parser.add_argument('--n', default=UNROLL_STEPS, type=int, help='Number of steps to unroll Bellman update')
  args = parser.parse_args()
  device = torch.device('cuda' if args.cuda else 'cpu')

  env = gym.make(params['env_name'])
  env = ptan.common.wrappers.wrap_dqn(env) # use the same wrappers as Chapter 6

  writer = SummaryWriter(comment='-' + params['run_name'] + '-%d-step' % args.n)

  net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
  target_net = ptan.agent.TargetNet(net) # make duplicate network to use in Bellman update
  selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
  epsilon_tracker = common.EpsilonTracker(selector, params)
  agent = ptan.agent.DQNAgent(net, selector, device=device)

  experience_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'],
                                                                 steps_count=args.n)
  buffer = ptan.experience.ExperienceReplayBuffer(experience_source,
                                                   buffer_size=params['replay_size'])
  optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

  # wrap main loop in a context manager that just keeps track of performance with tensorboard
  frame_index = 0
  with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
    while True:
      frame_index += 1
      buffer.populate(1) # makes agent interact with env 1 time and adds results to buffer

      epsilon_tracker.frame(frame_index)

      new_rewards = experience_source.pop_total_rewards()
      if new_rewards:
        if reward_tracker.reward(new_rewards[0], frame_index, selector.epsilon):
          break # break once our average reward is past a high threshold set in params

      if len(buffer) < params['replay_initial']:
        continue # don't start training until our buffer is full, or at least full enough

      # now train dqn
      optimizer.zero_grad()
      batch = buffer.sample(params['batch_size'])
      loss_value = common.calc_loss_dqn(batch, net, target_net.target_model, 
                                        gamma=params['gamma']**args.n, device=device)
      loss_value.backward()
      optimizer.step()

      # periodically update weights of target_net with net
      if frame_index % params['target_net_sync'] == 0:
        target_net.sync()