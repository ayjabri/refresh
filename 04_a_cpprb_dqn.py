#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ayman
"""
import os
import ptan
import torch
import argparse
import numpy as np
from datetime import datetime
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from lib import data, utils, model
from torch.utils.tensorboard import SummaryWriter


def calc_loss_dqn(batch, net, tgt_net, gamma, device, priority=False):
    """Return batch loss and priority updates (priority must be True) """

    if priority:
        states, actions, rewards, last_states, dones, weights, indexes = batch.values()
    else:
        states, actions, rewards, last_states, dones = batch.values()

    states_v = torch.FloatTensor(states).to(device)
    rewards_v = torch.FloatTensor(rewards).squeeze(-1).to(device)

    q_s_a = net(states_v)[range(len(actions)), actions.squeeze(-1)]
    with torch.no_grad():
        last_states_v = torch.FloatTensor(last_states).to(device)
        best_next_q_v = tgt_net.target_model(last_states_v).max(dim=1)[0]
        best_next_q_v[dones.squeeze(-1)] = 0.0
        exp_q_s_a = rewards_v + gamma * best_next_q_v.detach()
    if priority:
        weights_v = torch.FloatTensor(weights).to(device)
        l = (q_s_a - exp_q_s_a)**2
        loss_v = l * weights_v
        return loss_v.mean(), (loss_v + 1e-5).data.cpu().numpy(), indexes
    return torch.nn.functional.mse_loss(q_s_a, exp_q_s_a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Train on GPU')
    parser.add_argument('-p', '--priority', action='store_true',
                        help='Use Priority Replay Buffer')
    parser.add_argument('-s', '--steps', default=4, type=int,
                        help='steps to skip while training')
    parser.add_argument('-n', '--envs', default=3, type=int,
                        help='Number of environments to run simultaneously')
    parser.add_argument('-g', '--game', default='invaders',
                        help='OpenAI gym environment name')
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'

    params = data.params[args.game]

    envs = utils.createEnvs(params)

    shape = envs[0].observation_space.shape
    actions = envs[0].action_space.n

    net = model.DDQN(shape, actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.eps_start)
    eps_tracker = ptan.actions.EpsilonTracker(
        selector, params.eps_start, params.eps_final, params.eps_frames*args.envs)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=args.steps)

    env_dict = {'state': {'shape': shape, 'dtype': np.uint8},
                'action': {'dtype': np.int8},
                'reward': {},
                'last_state': {'shape': shape, 'dtype': np.uint8},
                'done': {'dtype': np.bool}
                }

    beta = 0.4
    buffer = PrioritizedReplayBuffer(params.buffer_size, env_dict) if args.priority else \
        ReplayBuffer(params.buffer_size, env_dict=env_dict)

    folder, sub_folder, log_dir = utils.writerDir(envs[0], args.steps)
    comment = "".join(
        [envs[0].game, '_', str(args.steps), '_', str(args.envs)])
    writer = SummaryWriter(comment=comment)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
                                                           cooldown=20000, verbose=True, min_lr=1e-7)
    mean = None
    best_reward = -float('inf')
    st = datetime.now()

    print(net)
    print(
        5*'*', f' Training {envs[0].game}: {device}/{args.envs} environments/{args.steps} steps', '*'*5)

    with ptan.common.utils.RewardTracker(writer) as tracker:
        for frame, exp in enumerate(exp_source):
            last_state = exp.state if exp.last_state is None else exp.last_state
            done = exp.last_state is None
            buffer.add(state=exp.state, action=exp.action,
                       reward=exp.reward, last_state=last_state, done=done)
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
                        if args.play:
                            utils.play(params.env, agent, wait=0.01)
                        break

            if frame < params.init_replay:
                continue

            if frame % args.envs == 0:
                batch = buffer.sample(params.batch_size, beta)
                # beta += (1-beta)/frame
                optimizer.zero_grad()
                if args.priority:
                    loss_v, batch_prios, batch_indexes = calc_loss_dqn(batch, net,
                                                                       tgt_net, params.gamma, device, True)
                else:
                    loss_v = calc_loss_dqn(
                        batch, net, tgt_net, params.gamma**args.steps, device=device)
                loss_v.backward()
                optimizer.step()
                if args.priority:
                    buffer.update_priorities(batch_indexes, batch_prios)
                # del batch, loss

                if mean and selector.epsilon <= params.eps_final:
                    scheduler.step(round(mean)*2/2)

                if frame % params.sync_nets == 0:
                    tgt_net.sync()
