#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempting to solve Atari games using A2C

@author: Ayman Jabri
"""
import torch
import ptan
from lib import data, utils, model
from tensorboardX import SummaryWriter

if __name__=='__main__':
    message = '*'*10 + '  A2C on Atari ' +'*'*10
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
    agent = ptan.agent.ActorCriticAgent(net, device=device, apply_softmax=True)

    exp_src = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma,steps_count=params.steps)
    generator = utils.BatchGenerator(exp_src, params)
    monitor = utils.MeanRewardsMonitor(envs[0], net, 'A2C', params.solve_rewards)

    writer = SummaryWriter(logdir=monitor.runs_dir,comment=params.frame_stack)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
    #                                                        cooldown=20000, verbose=True, min_lr=params.min_lr)

    print(net)
    print(5*'*', f' Training {envs[0].game}: {device}/{params.n_envs} environments/{params.steps} steps', '*'*5)
    
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for batch in generator:
            mean = generator.mean()
            if mean:    
                tracker.reward(reward, frame)
            if monitor(mean):
                break
            states, actions, q_refs = utils.unpack_a2c_batch(batch, net, params, device)
            
            optimizer.zero_grad()
            value_loss, policy_loss, entropy_loss = utils.calc_loss_a2c(batch, net, params, device=device)
            total_loss = value_loss + policy_loss + entropy_loss
            total_loss.backward()
            optimizer.step()
        


