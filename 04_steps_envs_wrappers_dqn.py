#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 08:45:18 2021

@author: ayman
"""
import os
import ptan

import torch
import argparse

from tensorboardX import SummaryWriter
from datetime import datetime
from lib import data, utils, model, play


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Train on GPU')
    parser.add_argument('-s', '--steps', default=4, type=int,
                        help='steps to skip while training')
    parser.add_argument('-n', '--envs', default=3, type=int,
                        help='Number of environments to run simultaneously')
    parser.add_argument('-g', '--game', default='invaders',
                        help='OpenAI gym environment name')
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'

    params = data.params[args.game]
    envs = utils.createLightWrapEnv(params.env, args.envs, 4)

    shape = envs[0].observation_space.shape
    actions = envs[0].action_space.n

    net = model.DDQN(shape, actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.eps_start)
    eps_tracker = ptan.actions.EpsilonTracker(
        selector, params.eps_start, params.eps_final, params.eps_frames)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=args.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.buffer_size)

    folder, sub_folder, log_dir = utils.writerDir(envs[0], args.steps)
    comment = "".join(
        [envs[0].game, '_', str(args.steps), '_', str(args.envs)])
    writer = SummaryWriter(comment=comment)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
                                                           cooldown=20000, verbose=True, min_lr=1e-7)
    frame = 0
    mean = None
    best_reward = -float('inf')
    st = datetime.now()

    print(net)
    print(
        5*'*', f' Training {envs[0].game}: {device}/{args.envs} environments/{args.steps} steps', '*'*5)
    buffer.populate(params.init_replay)
    exp_source.pop_rewards_steps()

    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += 1
            buffer.populate(args.envs)
            eps_tracker.frame(frame)
            reward = exp_source.pop_total_rewards()
            if reward:
                mean = tracker.reward(
                    reward[0], frame, epsilon=selector.epsilon)

                if mean:
                    if int(mean) > best_reward:
                        best_reward = int(mean)
                        torch.save(net.state_dict(), f=os.path.join(
                            folder, sub_folder, str(best_reward) + '.dat'))
                    if mean > params.solve_rewards:
                        print('Solved in {}!'.format(datetime.now()-st))
                        break
                # if mean and mean > params.solve_rewards:
                #     print('Solved in {}!'.format(datetime.now()-st))
                #     torch.save(net.state_dict(), f = os.path.join(folder,sub_folder,'solved.dat'))
                #     break

            batch = buffer.sample(params.batch_size)
            optimizer.zero_grad()
            loss = utils.calc_loss_dqn(
                batch, net, tgt_net, params.gamma**args.steps, device=device)
            loss.backward()
            optimizer.step()

            # del batch, loss

            if mean and selector.epsilon <= params.eps_final:
                scheduler.step(round(mean)*2/2)

            if frame % params.sync_nets == 0:
                tgt_net.sync()
