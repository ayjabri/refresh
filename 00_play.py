#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:39:16 2021

@author: ayman
"""
from ptan.actions import ArgmaxActionSelector
from ptan.agent import DQNAgent
import gym
import argparse
from torch import load
from lib import data, utils, model, atari_wrappers


GAMES = list(data.params.keys())


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=GAMES, help='choose from a list of trained games')
    parser.add_argument('--render', action='store_true', help='Show the game')
    parser.add_argument('--wait', type=float, default=0.0, help='pause for a little bit while playing')
    parser.add_argument('--stack', type=int, default=4, help='must match the number of frames stacked during training')
    parser.add_argument('--skip', type=int, help='Frames to skip when stacking. Must specifiy when selecting --lw')
    parser.add_argument('--model', type=str, help='Path to the trained state dict model')
    parser.add_argument('--lw', action='store_true', help='Use light wrapper to create the environment. Must select this if the network was trained using light wrapper')
    parser.add_argument('--record', action='store_true', help='record a video of the game and store it in ~/Videos')
    args = parser.parse_args()

    if args.lw: assert isinstance(args.skip, int)

    params = data.params[args.env]
    params.max_steps = None
    env = utils.createEnvs(params, stack_frames=args.stack, episodic_life=False,
                           reward_clipping=False)[0]
    if args.lw:
        env = gym.make(params.env)
        env = atari_wrappers.wrap_dqn_light(env, stack_frames=args.stack, skip=args.skip,
                                            episodic_life=False, reward_clipping=False)
    # record a video of the game
    if args.record: gym.wrappers.Monitor(env, "Videos", force=True)

    shape, actions = env.observation_space.shape, env.action_space.n
    net = model.DDQN(shape, actions)
    if args.model: net.load_state_dict(load(args.model, map_location='cpu'))

    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector)

    utils.play(env, agent, wait=args.wait, render=args.render)

