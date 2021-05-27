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
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from . import atari_wrappers


def unpack_batch(batch):
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
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def createEnvs(params, stack_frames=4, episodic_life=True, reward_clipping=True):
    """Create an OpenAI gym environments wrapped and seed set"""
    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env)
        env = ptan.common.wrappers.wrap_dqn(env,stack_frames, episodic_life, reward_clipping)
        if params.seed: env.seed(params.seed)
        envs.append(env)
    return envs


def bad_wrap_dqn(params):
    """ Very bad wrappers that eat up the memory up to 100%! I don't know why """
    envs = []
    for _ in range(params.n_envs):
        env = atari_wrappers.make_atari(
            params.env, skip_noop=True, skip_maxskip=True)
        env = atari_wrappers.wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=True,
                                            pytorch_img=True, frame_stack_count= 4, skip_firereset=True)
        if params.seed:
            env.seed(params.seed)
        envs.append(env)
    return envs


def writerDir(env, steps):
    folder = (env.game + '_' + "DQN").capitalize()
    sub_folder = datetime.now().strftime('%h_%d_%Y_%H_%M_')+str(steps)+'_steps'

    if not os.path.exists(os.path.join(folder, sub_folder)):
        os.makedirs(os.path.join(folder, sub_folder))

    log_dir = os.path.join('runs', sub_folder)
    return folder, sub_folder, log_dir


def play(game, agent=None, wait=0.0, render=True):
    if not isinstance(game, gym.Env):
        env = createEnvs(game,stack_frames=4, episodic_life=False, reward_clipping=False)[0]
    else:
        env = game
    state = env.reset()
    rewards = 0
    while True:
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
            print('Finished playing with {} rewards'.format(rewards))
            break
    env.close()
