#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import ptan
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from lib.data import params
from lib import model,utils


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',default=True, action='store_true',help='Use GPU in training')
    parser.add_argument('--env',default='pong', type=str,help='OpenAI environment name')
    parser.add_argument('--steps', default=4, type=int, help='Skip N steps. Adjusts gamma by gamma^N')
    args = parser.parse_args()
    
    device = 'cuda' if args.cuda else 'cpu'
    params = params[args.env]
    
    N0=24
    np.random.seed(N0)
    torch.manual_seed(N0)
    env = gym.make(params.env)
    env.seed(N0)
    env = ptan.common.wrappers.wrap_dqn(env)
    
    shape = env.observation_space.shape
    actions = env.action_space.n
    net = model.DQNetConv(shape,actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.eps_start)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, 
                                              params.eps_final, params.eps_frames)
    
    process = utils.LazyFrameProcessor
    agent = ptan.agent.DQNAgent(net, selector,device=device,
                                preprocessor=ptan.agent.default_states_preprocessor)
    exp_source= ptan.experience.ExperienceSourceFirstLast(env, agent, params.gamma**args.steps, steps_count=args.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)
    
    comment = utils.writerComment(env)
    writer = SummaryWriter(comment=comment+'_'+args.steps)
    
    state = utils.LazyFrameProcessor(env.reset()).to(device)
    writer.add_graph(net,state)
    print(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    frame = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += 1
            eps_tracker.frame(frame)
            buffer.populate(1)
            reward = exp_source.pop_total_rewards()
            if reward:
                mean = tracker.reward(reward[-1], frame, epsilon=selector.epsilon)
                if mean and mean > params.solve_rewards:
                    print('Solved!')
                    break            
            if len(buffer) < params.init_replay:
                continue      
            
            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size)
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net, params.gamma**args.steps, device)
            loss_v.backward()
            optimizer.step()
            writer.add_scalar('DQN Loss', loss_v.item())
            
            del batch, loss_v
            
            if frame % params.sync_nets ==0:
                tgt_net.sync()
            
            
