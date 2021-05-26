#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:42:29 2021

@author: ayman
"""

# -*- coding: utf-8 -*-

import ptan
from collections import namedtuple
from . import utils
EpisodeEnd = namedtuple('EpisodeEnd', ['step', 'reward', 'epsilon'])


def data_fun(net, exp_queue, params, device='cpu'):
    """
    Definition: data_fun(net,exp_queue,ENV_ID,STEPS=1,N_ENVS=1)

    Stores ptan FirstLast experiences in a multiprocess Queue()

    Parameters
    ----------
    net : Deep-Q Neural Netwok class
        Can be any DQN. Tested with DuelDQN network

    exp_queue : Pytorch Multiprocessing.Queue()
        Shared Queue to store experiences.

    params : a simple name space dict that contains hyperparameters

    Returns
    -------
    Stores experiences in a multiprocessing Queue(). It also stores step,reward and epsilon
    as named tuple (EndEpisode) at the end of each episode.

    Use as target for Multiprocessing.  

    N-Environments:
    --------
    To use N number of environments you must do the following changes to your training loop:
        1- Use common SEED in all environments        

        2- Multiply batch-size by N        

        3- Multipy frame by N in Epsilon tracker.frame() function if using one      

        4- Multiply fps by N (haven't tried it yet!)       

        5- Populate N steps if using Buffer       
    """
    envs = utils.createLightWrapEnv(params.env, params.n_envs, 4, 124)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, params.eps_final,
                                              params.eps_frames)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           params.gamma, steps_count=params.steps)
    step = 0
    for exp in exp_source:
        step += 1
        eps_tracker.frame(step*params.n_envs)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(step, new_reward[0], selector.epsilon))
        exp_queue.put(exp)


def data_fun_global(net, exp_queue, params, frames, episodes, device='cpu'):
    envs = utils.createLightWrapEnv(params.env, 1, 4, 124)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, params.eps_final,
                                              params.eps_frames)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           params.gamma, steps_count=params.steps)
    for exp in exp_source:
        frames.value += 1
        step = frames.value
        eps_tracker.frame(step*params.n_envs)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episodes.value += 1
            exp_queue.put(EpisodeEnd(step, new_reward[0], selector.epsilon))
        exp_queue.put(exp)


class MPBatchGenerator(object):
    """
    Yields batchs from experiences stored in multiprocess Queue()


    Parameters:
    -------
    buffer: ptan.experience.ExperienceReplayBuffer(exp_source=None)
        Buffer object that will store FirstLast experiences

    exp_queue: Torch Multiprocessing Queue()
        Queue of specific size the will store observations and end of episode readings

    initial: Int
        Number of stored experiences before start sampling

    batch_size: int
        The size of batch to generate

    multiplier: int. Defaults to 1
        Multiply batch size by this number
    """

    def __init__(self, buffer, exp_queue, initial, batch_size, multiplier):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.initial = initial
        self.batch_size = batch_size
        self.multiplier = multiplier
        self._total_rewards = []
        self.frame = 0
        self.episode = 0
        self.epsilon = 0.0

    def pop_rewards_idx_eps(self):
        res = list(self._total_rewards)
        self._total_rewards.clear()
        return res

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        while True:
            while not self.exp_queue.empty():
                exp = self.exp_queue.get()
                if isinstance(exp, EpisodeEnd):
                    self._total_rewards.append(exp.reward)
                    self.frame += exp.step
                    self.epsilon = exp.epsilon
                    self.episode += 1
                else:
                    self.buffer._add(exp)
                    self.frame += 1
                # del exp
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size * self.multiplier)
