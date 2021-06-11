#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempting to solve Atari games using A2C

@author: Ayman Jabri
"""
import os
import torch
import ptan
from lib import data, utils, mp_utils, model
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp


ALGORITHM = 'A3C'
FORKS = 8
MINI_BATCH = 16


if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    message = '*'*10 + '  A3C on Atari ' +'*'*10
    args = utils.argpars_dqn(message)
    params = data.params[args.env]
    utils.update_params(params, args)

    params.n_envs = max(params.n_envs, 8)

    device = 'cuda' if args.cuda else 'cpu'
    envs = utils.createEnvs(params, stack_frames=2)
    shape = envs[0].observation_space.shape
    actions = envs[0].action_space.n
    net = model.A2CNet(shape, actions)
    net.to(device)
    net.share_memory()
    
    # agent = ptan.agent.ActorCriticAgent(net, device=device, apply_softmax=True)

    # exp_src = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma,steps_count=params.steps)
    # generator = utils.BatchGenerator(exp_src, params)

    mean_monitor = utils.MeanRewardsMonitor(envs[0], net, ALGORITHM, params.solve_rewards)

    writer = SummaryWriter(logdir=mean_monitor.runs_dir,comment=params.frame_stack)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
    #                                                        cooldown=20000, verbose=True, min_lr=params.min_lr)

    
    print('# Parameters: ', utils.count_parameters(net))
    print(net)
    print('*'*10, ' Start Training ',
          envs[0].game, ' {} '.format(device), '*'*10)
    

    frame = 0
    episode = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for batch in generator:
            reward = generator.pop_total_rewards()
            if reward:
                episode += 1
                mean = tracker.reward(
                    reward[0], generator.frame)
                if mean_monitor(mean):
                    break
            
            #### Training ###
            states, actions, q_refs = utils.unpack_a2c_batch(batch, net, params, device)
            
            optimizer.zero_grad()
            value_loss, policy_loss, entropy_loss = utils.calc_loss_a2c(batch, net, params, device=device)
            total_loss = value_loss + policy_loss + entropy_loss
            total_loss.backward()
            optimizer.step()
        


