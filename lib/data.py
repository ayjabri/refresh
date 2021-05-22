#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:15:23 2021

@author: ayman
"""

from types import SimpleNamespace



params = {'pong':SimpleNamespace(**{'env':'PongNoFrameskip-v4',
                                    'lr':1e-4,
                                    'init_replay':10_000,
                                    'eps_start':1.0,
                                    'eps_final':0.02,
                                    'eps_frames':50_000,
                                    'buffer_size':100_000,
                                    'gamma':0.97,
                                    'batch_size':32,
                                    'solve_rewards':17,
                                    'sync_nets':1000,
                                    'seed':124,
                                    'n_envs':3,
                                    }),
         'breakout':SimpleNamespace(**{'env':'BreakoutNoFrameskip-v4',
                                    'lr':1e-4,
                                    'init_replay':25_000,
                                    'eps_start':1.0,
                                    'eps_final':0.1,
                                    'eps_frames':150_000,
                                    'buffer_size':250_000,
                                    'gamma':0.999,
                                    'batch_size':32,
                                    'solve_rewards':100,
                                    'sync_nets':1000,
                                    'seed':124,
                                    'n_envs':3,
                                    }),
         'invaders':SimpleNamespace(**{'env':'SpaceInvadersNoFrameskip-v4',
                                    'lr':1e-4,
                                    'init_replay':10_000,
                                    'eps_start':1.0,
                                    'eps_final':0.02,
                                    'eps_frames':25_000,
                                    'buffer_size':100_000,
                                    'gamma':0.99,
                                    'batch_size':16,
                                    'solve_rewards':450,
                                    'sync_nets':1000,
                                    'seed':124,
                                    'n_envs':4,
                                    }),
         
         }

