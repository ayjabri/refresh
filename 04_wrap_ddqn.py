#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:12:09 2021

@author: ayman
"""

import gym
import ptan
import torch
import argparse
from tensorboardX import SummaryWriter
from lib import utils, model, data, atari_wrappers


ALGORITHM = 'DDQN_WRAP'
GAMES = list(data.params.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='pong', choices=GAMES,
                        help='OpenAI gym environment name.\n Default: pong')
    parser.add_argument('--cuda', action='store_true',
                        help='Activate GPU in training')
    args = parser.parse_args()

    params = data.params[args.env]

    torch.manual_seed(params.seed)
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # We're using light wrappers that stacks 2 frames and using PIL-smid library
    # Accordign to MaxLapan this should improve both fps and convergance
    env_ = gym.make(params.env)
    env = atari_wrappers.wrap_dqn_light(env_, stack_frames=2, skip=6)

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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='max', patience=10_000, verbose=True,
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
                    reward[-1], frame, epsilon=selector.epsilon)
                if mean_monitor(mean):
                    break

            if len(buffer) < params.init_replay:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size)
            loss_v = utils.calc_loss_dqn(
                batch, net, tgt_net, params.gamma**params.steps, device)
            loss_v.backward()
            optimizer.step()

            if mean and selector.epsilon <= params.eps_final:
                lr_scheduler.step(mean_monitor.best_reward)
                writer.add_scalar(
                    'LearningRate', scalar_value=lr_scheduler._last_lr, global_step=frame)
            del batch, loss_v
            if frame % 1000 == 0:
                tgt_net.sync()
