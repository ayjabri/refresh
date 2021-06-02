#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:54:31 2021

@author: ayman
"""
import os
import gym
import ptan
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from . import atari_wrappers

# =============================================================================
# Classes:
#   1- Mean Rewards Monitor: tracks mean rewards and saves when a higher mean rewards is achived
#
# =============================================================================

class MeanRewardsMonitor:
    """Track mean rewards, save model each new reward"""
    def __init__(self, env, net, algorithm, solve_rewrads, path='../data'):

        self.env = env
        self.net = net

        self.path = path
        self.algorithm = algorithm
        self.solve_rewrads = solve_rewrads

        self.start = datetime.now()
        self.best_reward = -float('inf')
        self.get_runs_save_dir()

    def get_runs_save_dir(self):
        fsub = datetime.now().strftime('%m%d%Y_%H%M')
        self.runs_dir = os.path.join(self.path, self.env.game, 'runs', fsub)
        self.save_dir = os.path.join(
            self.path, self.env.game, 'models', self.algorithm, fsub)

        if not os.path.exists(self.runs_dir):
            os.makedirs(self.runs_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, mean):
        if mean:
            if mean > self.best_reward:
                self.best_reward = int(mean)
                save_path = os.path.join(
                    self.save_dir, str(self.best_reward) + '.dat')
                torch.save(self.net.state_dict(), f=save_path)
            if mean > self.solve_rewrads:
                duration = datetime.now()-self.start
                print(f'Solved in {duration}')
                return True
        return False


# =============================================================================
# Support functions for DQN training
# =============================================================================


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
    # Q_values = net(states_v)
    # state_action_values = Q_values[range(len(actions)), actions]
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)



def createEnvs(params, stack_frames=4, episodic_life=True, reward_clipping=True):
    """
    Return n numberof OpenAI gym environments wrapped and seed set.
    params: dictionary of the environment's hyperparameters
    stack_frames: number of frame to stack in each observation
    episodic_life: for games with multiple lives
    reward_clipping: normalize rewards to 1 and 0
    """

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env)
        env = ptan.common.wrappers.wrap_dqn(
            env, stack_frames, episodic_life, reward_clipping)
        if params.max_steps is not None:
            env = atari_wrappers.TimeLimit(env, params.max_steps)
        if params.seed:
            env.seed(params.seed)
        envs.append(env)
    return envs


@torch.no_grad()
def play(game, agent=None, wait=0.0, render=True):
    """
    Play and episode of OpenAI gym

    Parameters
    ----------
    game : gym.Env or Str
        If string it will create a new environment using CreateEnvs function.
    agent : ptan.agent, optional
        Trained agent. If None it will select random actions from the action space. The default is None.
    wait : Int, optional
        Time to pause after each render in case the play was too fast for human eyes. The default is 0.0.
    render : Bool, optional
        Show the game. The default is True.

    Returns
    -------
    None.

    """
    if not isinstance(game, gym.Env):
        env = createEnvs(game, stack_frames=4,
                         episodic_life=False, reward_clipping=False)[0]
    else:
        env = game
    state = env.reset()
    rewards = 0
    steps = 0
    while True:
        steps += 1
        if render:
            env.render()
            time.sleep(wait)
        if agent is None:
            action = env.action_space.sample()
        else:
            state = np.expand_dims(state, 0)
            action = agent(state)[0]
        state, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            print('Done!\nPlayed {} steps and got {} rewards'.format(steps, rewards))
            break
    env.close()



def argpars_dqn():
    """Standard arg parse to use in DQN trainings"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',   action='store_true', help='Train on GPU')
    parser.add_argument('--env',    default='pong', help='OpenAI gym environment name')
    parser.add_argument('--lr',     type=int, help='Learning Rate')
    parser.add_argument('--batch',  type=int, help='Batch size')
    parser.add_argument('--n_envs', type=int, help='Number of environments to run simultaneously')
    parser.add_argument('--stack',  default=2, type=int, help='Frames to stack in each observation')
    parser.add_argument('--steps',  type=int, help='steps to skip while training')
    parser.add_argument('--max',    type=int, help='Maximum steps of episodes')
    parser.add_argument('--skip',   default=6, type=int, help='Frames to skip when stacking. Default=6')
    parser.add_argument('--resnet', action='store_true', help='Use ResNet DDQ network in training')

    return parser.parse_args()


def update_params(params, args):
    """Update parameters based on user input(used after argparse_dqn function)."""

    if args.lr      is not None: params.lr = args.lr
    if args.batch   is not None: params.batch_size = args.batch
    if args.n_envs  is not None: params.n_envs = args.n_envs
    if args.stack   is not None: params.frame_stack = args.stack
    if args.steps   is not None: params.steps = args.steps
    if args.max     is not None: params.max_steps = args.max
