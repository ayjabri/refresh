#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:39:16 2021

@author: ayman
"""
from ptan.actions import ArgmaxActionSelector
from ptan.agent import DQNAgent

import argparse
from torch import load
from lib import data, utils, model


GAMES = list(data.params.keys())


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=GAMES, help='choose from a list of trained games')
    parser.add_argument('--render', action='store_true', help='Show the game')
    parser.add_argument('--wait', type=float, default=0.0, help='pause for a little bit while playing')
    parser.add_argument('--stack', type=int, default=4, help='must match the number of frames stacked during training')
    parser.add_argument('--model', type=str, help='Path to the trained state dict model')
    args = parser.parse_args()


    params = data.params[args.env]
    params.max_steps = None
    env = utils.createEnvs(params, stack_frames=args.stack, episodic_life=False,
                           reward_clipping=False)[0]

    shape, actions = env.observation_space.shape, env.action_space.n
    net = model.DDQN(shape, actions)
    if args.model: net.load_state_dict(load(args.model, map_location='cpu'))

    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector)

    utils.play(env, agent, wait=args.wait, render=args.render)

