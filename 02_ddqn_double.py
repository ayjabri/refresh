#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Double DQN.

@author:Ayman Jabri
"""
import sys

import ptan
import argparse
import gym

import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from lib import utils, data, model


ALGORITHM = 'DOUBLE_DDQN'
GAMES = list(data.params.keys())


def calc_double_dqn_loss(batch, net, tgt_net, gamma, device='cpu'):
    """
    Loss function implementation of DeepMind paper.
    
    [Deep Reinforcement Learning with Double Q-Learning
    ([3] van Hasselt, Guez, and Silver, 2015)]
    """
    states, actions, rewards, dones, last_states = utils.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    rewards_v = torch.tensor(rewards).to(device)

    q_state_action_v = net(states_v)[range(len(actions)), actions]
    with torch.no_grad():
        last_states_v = torch.tensor(last_states).to(device)
        next_actions_v = net(last_states_v).argmax(dim=1)
        next_q_state_action_v = tgt_net.target_model(
            last_states_v)[range(len(actions)), next_actions_v]
        next_q_state_action_v[dones] = 0.0
    exp_state_action_v = rewards_v + gamma * next_q_state_action_v
    return F.mse_loss(q_state_action_v, exp_state_action_v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='pong', choices=GAMES,
                        help='OpenAI gym environment name.\n Default: pong')
    parser.add_argument('--cuda', action='store_true',
                        help='Activate GPU in training')
    parser.add_argument('--double', action='store_true',
                        help='calculate loss using doulbe algorithm')

    args = parser.parse_args()

    if not args.double:
        sys.exit("Invalid selection. You must add --double flag to run this script")

    params = data.params[args.env]
    torch.manual_seed(params.seed)
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    env = gym.make(params.env)
    env = ptan.common.wrappers.wrap_dqn(env)

    shape = env.observation_space.shape
    actions = env.action_space.n

    net = model.DDQN(shape, actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start,
                                              params.eps_final, params.eps_frames)

    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_src = ptan.experience.ExperienceSourceFirstLast(
        env, agent, params.gamma, steps_count=1)

    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_src, params.buffer_size)

    mean_monitor = utils.MeanRewardsMonitor(
        env, net, ALGORITHM, params.solve_rewards)

    writer = SummaryWriter(logdir=mean_monitor.runs_dir,
                           comment=params.frame_stack)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10_000, verbose=True,
                                                              factor=0.75, min_lr=params.min_lr, cooldown=10_000)
    print(net)
    print('*'*10, ' Start Training ',
          env.game, ' {} '.format(device), '*'*10)

    frame = 0
    episode = 0

    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += 1
            eps_tracker.frame(frame)
            buffer.populate(1)
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
            loss_v = calc_double_dqn_loss(
                batch, net, tgt_net, params.gamma**params.steps, device)
            loss_v.backward()
            optimizer.step()

            if mean and selector.epsilon <= params.eps_final:
                lr_scheduler.step(mean_monitor.best_reward)
                writer.add_scalar(
                    'LearningRate', scalar_value=lr_scheduler._last_lr, global_step=frame)

            if frame % 1000 == 0:
                tgt_net.sync()
