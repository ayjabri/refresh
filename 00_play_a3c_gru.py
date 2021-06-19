#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play atari using a neural network with GRU Cell


Created on Thu Jun 17 05:55:25 2021

@author: ayman
"""

from ptan.actions import ProbabilityActionSelector
from ptan.agent import ActorCriticAgent
import gym
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
    parser.add_argument('--skip', default=4, type=int, help='Frames to skip when stacking. Must specifiy when selecting --lw')
    parser.add_argument('--model', type=str, help='Path to the trained state dict model')
    parser.add_argument('--record', action='store_true', help='record a video of the game and store it in ~/Videos')
    args = parser.parse_args()

    

    params = data.params[args.env]
    params.max_steps = None
    env = utils.createEnvs(params, stack_frames=args.stack, episodic_life=False,
                           reward_clipping=False, skip=args.skip)[0]
    # recording the game
    if args.record: env = gym.wrappers.Monitor(env, "Videos", force=True)

    shape, actions = env.observation_space.shape, env.action_space.n
    net = model.A2Cgru(shape, actions)
    print(net)
    if args.model: net.load_state_dict(load(args.model, map_location='cpu'))

    selector = ProbabilityActionSelector()
    agent = ActorCriticAgent(net, selector, apply_softmax=True)

    utils.play(env, agent, wait=args.wait, render=args.render)
