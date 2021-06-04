# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

https://arxiv.org/abs/1507.06527
https://arxiv.org/pdf/1507.06527.pdf

"""
import gym
import ptan
import torch
import torch.nn as nn
import numpy as np
from lib import utils, atari_wrappers, data
from tensorboardX import SummaryWriter

ALGORITHM = "DQRNN"


class DRQNet(nn.Module):
    def __init__(self, shape, actions, conv_features=64, num_layers=4, hidden_size=128, device='cuda'):
        super(DRQNet, self).__init__()

        self.shape = shape
        self.actions = actions
        self.conv_features = conv_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.conv = nn.Sequential(nn.Conv2d(shape[0], 32, kernel_size=8, stride=4, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(32, conv_features, kernel_size=4, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(conv_features, conv_features,
                                            kernel_size=3, stride=1, bias=False),
                                  nn.ReLU(),
                                  nn.AdaptiveMaxPool2d((conv_features, 1))
                                  )

        self.gru_input = self._gru_input_size()

        self.gru = nn.GRU(self.gru_input, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.act = nn.ReLU()

        fc_input_size = hidden_size * conv_features

        self.fc = nn.Sequential(nn.Flatten(1),
                                nn.Linear(fc_input_size, actions))
        self.to(device)

    def forward(self, x, hidden):
        """
        GRU <- (batch, h, w)
        h0: (layers, batch, hidden_size)
        """
        fx = x.float()/255
        out = self.conv(fx)
        # gru_input: (batch, h, w), h0: (layers, batch, hidden_size)
        self.gru.flatten_parameters()
        out, new_hidden = self.gru(out[:, :, :, -1], hidden)
        out = self.fc(self.act(out))
        return out, new_hidden

    def _gru_input_size(self):
        o = torch.zeros(1, *self.shape)
        return np.product(self.conv(o).detach().numpy().shape[-2:])

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


def unpack_batch(batch):
    """Unpack standard ptan experience first-last batch."""
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    """Calculate DeepQ Loss """
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones).to(device)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    message = '*** Using Deep Recurrent Q-Learning Algorithm *** '
    args = utils.argpars_dqn(message)
    params = data.params[args.env]
    utils.update_params(params, args)

    device = 'cuda' if args.cuda else 'cpu'
    # 'RiverraidNoFrameskip-v4' CentipedeNoFrameskip-v4 'PongNoFrameskip-v4'

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env)
        env = atari_wrappers.wrap_dqn_light(env, stack_frames=params.frame_stack,
                                            skip=args.skip, max_episode_steps=args.max)
        env.seed(params.seed)
        envs.append(env)

    shape = env.observation_space.shape
    actions = env.action_space.n

    net = DRQNet(shape, actions, hidden_size=128, device=device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()

    agent = ptan.agent.DQNAgent(lambda x: net(x, net.init_hidden(params.n_envs))[0], selector, device=device)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, params.eps_final, params.eps_frames)

    exp_src = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=params.gamma, steps_count=params.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_src, params.buffer_size)

    mean_monitor = utils.MeanRewardsMonitor(env, net, ALGORITHM, params.solve_rewards)

    writer = SummaryWriter(logdir=mean_monitor.runs_dir,
                           comment=str(params.n_envs))

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr,)

    print(net)
    print('*'*10, ' Start Training ',
          env.game, ' {} '.format(device), '*'*10)

    frame = 0
    episode = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += params.n_envs
            eps_tracker.frame(frame)
            buffer.populate(params.n_envs)
            reward = exp_src.pop_total_rewards()
            if reward:
                episode += 1
                mean = tracker.reward(
                    reward[0], frame, epsilon=selector.epsilon)
                if mean_monitor(mean):
                    break

            if len(buffer) < params.init_replay:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size)
            loss_v = calc_loss_dqn(batch,
                                   lambda x: net(x, net.init_hidden(params.batch_size))[0],
                                   lambda x: tgt_net.target_model(x, net.init_hidden(params.batch_size))[0],
                                   params.gamma**params.steps, device=device)
            loss_v.backward()
            optimizer.step()

            if frame % params.sync_nets == 0:
                tgt_net.sync()
