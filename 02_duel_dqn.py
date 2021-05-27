#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 05:45:57 2021

@author: ayman
"""
import os
import ptan
import torch
import argparse
from tensorboardX import SummaryWriter
from lib import utils, model, data, atari_wrappers
from datetime import datetime

ALGORITHM = 'DDQN'


### Do NOT use this wrapper ###
def createLightWrapEnv(params):
    """
    Something is wrong with this wrapper! it causes the sytsem to eat up all 
    the available memory! I need to troubleshoot it when I have time
    """
    envs = []
    for _ in range(params.n_envs):
        env = atari_wrappers.make_atari(
            params.env, skip_noop=False, skip_maxskip=False)
        env = atari_wrappers.wrap_deepmind(env, clip_rewards=True, pytorch_img=True, frame_stack=True,
                                           frame_stack_count=params.frame_stack)
        if params.seed:
            env.seed(params.seed)
        envs.append(env)
    return envs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='pong',
                        help='OpenAI gym environment name.\n Default: pong')
    parser.add_argument('--cuda', action='store_true',
                        help='Activate GPU in training')
    parser.add_argument('--simple', action='store_true',
                        help='DO NOT SELECT THIS.. Use simple wrapper to enhance convergence speed')
    args = parser.parse_args()

    params = data.params[args.env]
    torch.manual_seed(params.seed)
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    envs = createLightWrapEnv(params) if args.simple else utils.createEnvs(params)

    shape = envs[0].observation_space.shape
    actions = envs[0].action_space.n

    net = model.DDQN(shape, actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start,
                                              params.eps_final, params.eps_frames)

    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_src = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=params.steps)

    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_src, params.buffer_size)

    folder = (envs[0].game + '_' + ALGORITHM).capitalize()
    sub_folder = datetime.now().strftime('%h_%d_%Y_%H_%M_') + \
        str(params.steps)+'_steps'

    if not os.path.exists(os.path.join(folder, sub_folder)):
        os.makedirs(os.path.join(folder, sub_folder))

    log_dir = os.path.join('runs', sub_folder)
    wrapper = 'simple_wrapper' if args.simple else 'normal_wrapper'
    writer = SummaryWriter(logdir=log_dir, comment=wrapper)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10_000, verbose=True,
                                                              factor=0.75, min_lr=1e-6, cooldown=10_000)
    print(net)
    print('*'*10, ' Start Training ',
          envs[0].game, ' {} '.format(device), '*'*10)

    best_reward = -float('inf')
    frame = 0
    episode = 0
    start_time = datetime.now()

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
                if mean:
                    if int(mean) > best_reward:
                        best_reward = int(mean)
                        save_time = datetime.now().strftime('%H_%M')
                        save_path = os.path.join(folder, sub_folder, str(
                            int(best_reward)) + '_' + save_time + '_.dat')
                        torch.save(net.state_dict(), f=save_path)
                    if mean > params.solve_rewards:
                        duration = datetime.now()-start_time
                        print(f'Solved in {duration}')
                        break

            if len(buffer) < params.init_replay:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size)
            loss_v = utils.calc_loss_dqn(
                batch, net, tgt_net, params.gamma**params.steps, device)
            loss_v.backward()
            optimizer.step()

            if mean:
                lr_scheduler.step(int(best_reward))
                writer.add_scalar(
                    'LearningRate', scalar_value=lr_scheduler._last_lr, global_step=frame)

            del batch, loss_v
            if frame % 1000 == 0:
                tgt_net.sync()
