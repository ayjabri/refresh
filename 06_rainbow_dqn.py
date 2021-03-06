#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 08:45:18 2021

@author: ayman
"""

import ptan
import gym
import torch


from tensorboardX import SummaryWriter

from lib import data, utils, model, atari_wrappers


ALGORITHM = 'RAINBOW_DDQN'
GAMES = list(data.params.keys())


if __name__ == '__main__':
    args = utils.argpars_dqn()
    params = data.params[args.env]
    utils.update_params(params, args)

    device = 'cuda' if args.cuda else 'cpu'

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env)
        env = atari_wrappers.wrap_dqn_light(env, params.frame_stack, args.skip)
        if params.seed:
            env.seed(params.seed)
        envs.append(env)

    shape = env.observation_space.shape
    actions = env.action_space.n

    if args.resnet:
        net = model.ResNetDDQN(shape, actions).to(device)
    else:
        net = model.DDQN(shape, actions).to(device)

    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.eps_start)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, params.eps_final, params.eps_frames)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma, steps_count=params.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)

    mean_monitor = utils.MeanRewardsMonitor(env, net, ALGORITHM, params.solve_rewards)

    writer = SummaryWriter(logdir=mean_monitor.runs_dir,comment=params.frame_stack)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
                                                           cooldown=20000, verbose=True, min_lr=params.min_lr)

    print(net)
    print(5*'*', f' Training {env.game}: {device}/{params.n_envs} environments/{params.steps} steps', '*'*5)

    # play init_replay steps before training
    buffer.populate(params.init_replay)
    exp_source.pop_rewards_steps()

    torch.autograd.set_detect_anomaly(True) ####

    frame = 0
    episode = 0
    mean = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += params.n_envs
            buffer.populate(params.n_envs)
            eps_tracker.frame(frame)
            reward = exp_source.pop_total_rewards()
            if reward:
                mean = tracker.reward(
                    reward[0], frame, epsilon=selector.epsilon)
                if mean_monitor(mean):
                    break

            batch = buffer.sample(params.batch_size * params.n_envs)
            optimizer.zero_grad()
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net, params.gamma**params.steps, device=device)
            loss_v.backward()
            optimizer.step()

            writer.add_scalar('TrainingLoss', loss_v.detach().item(), global_step=frame)

            if mean and selector.epsilon <= params.eps_final:
                lr_scheduler.step(mean_monitor.best_reward)
                writer.add_scalar('LearningRate', scalar_value=lr_scheduler._last_lr, global_step=frame)

            del batch, loss_v
            if frame % 1000 == 0:
                tgt_net.sync()
