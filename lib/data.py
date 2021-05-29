#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:15:23 2021

@author: ayman
"""

from types import SimpleNamespace


params = {'pong': SimpleNamespace(**{'env': 'PongNoFrameskip-v4',
                                     'lr': 1e-4,
                                     'init_replay': 10_000,
                                     'eps_start': 1.0,
                                     'eps_final': 0.02,
                                     'eps_frames': 50_000,
                                     'buffer_size': 100_000,
                                     'gamma': 0.99,
                                     'batch_size': 64,
                                     'solve_rewards': 17,
                                     'sync_nets': 1000,
                                     'seed': 144,
                                     'n_envs': 3,
                                     'frame_stack': 4,
                                     'steps': 4,
                                     'max_steps': None,
                                     'min_lr':1e-6,
                                     }),
          'breakout': SimpleNamespace(**{'env': 'BreakoutNoFrameskip-v4',
                                         'lr': 1e-4,
                                         'init_replay': 25_000,
                                         'eps_start': 1.0,
                                         'eps_final': 0.1,
                                         'eps_frames': 150_000,
                                         'buffer_size': 250_000,
                                         'gamma': 0.99,
                                         'batch_size': 32,
                                         'solve_rewards': 100,
                                         'sync_nets': 1000,
                                         'seed': 124,
                                         'n_envs': 3,
                                         'frame_stack': 2,
                                         'steps': 4,
                                         'max_steps': None,
                                         'min_lr':1e-5,
                                         }),
          'invaders': SimpleNamespace(**{'env': 'SpaceInvadersNoFrameskip-v4',
                                         'lr': 1e-4,
                                         'init_replay': 10_000,
                                         'eps_start': 1.0,
                                         'eps_final': 0.02,
                                         'eps_frames': 50_000,
                                         'buffer_size': 250_000,
                                         'gamma': 0.99,
                                         'batch_size': 128,
                                         'solve_rewards': 250,
                                         'sync_nets': 1000,
                                         'seed': 145,
                                         'n_envs': 2,
                                         'frame_stack': 2,
                                         'steps': 4,
                                         'max_steps': None,
                                         'min_lr':1e-6,
                                         }),
          'boxing': SimpleNamespace(**{'env': 'BoxingNoFrameskip-v4',
                                         'lr': 1e-4,
                                         'init_replay': 10_000,
                                         'eps_start': 1.0,
                                         'eps_final': 0.02,
                                         'eps_frames': 50_000,
                                         'buffer_size': 100_000,
                                         'gamma': 0.97,
                                         'batch_size': 16,
                                         'solve_rewards': 15,
                                         'sync_nets': 1000,
                                         'seed': 155,
                                         'n_envs': 4,
                                         'frame_stack': 4,
                                         'steps': 4,
                                         'max_steps': 300,
                                         'min_lr':1e-6,
                                         }),
          'bowling': SimpleNamespace(**{'env': 'BowlingNoFrameskip-v4',
                                         'lr': 1e-3,
                                         'init_replay': 10_000,
                                         'eps_start': 1.0,
                                         'eps_final': 0.02,
                                         'eps_frames': 50_000,
                                         'buffer_size': 100_000,
                                         'gamma': 0.97,
                                         'batch_size': 16,
                                         'solve_rewards': 15,
                                         'sync_nets': 1000,
                                         'seed': 155,
                                         'n_envs': 4,
                                         'frame_stack': 4,
                                         'steps': 4,
                                         'max_steps': 300,
                                         'min_lr':1e-5,
                                         }),
          'demon': SimpleNamespace(**{'env': 'DemonAttackNoFrameskip-v4',
                                         'lr': 1e-3,
                                         'init_replay': 10_000,
                                         'eps_start': 1.0,
                                         'eps_final': 0.02,
                                         'eps_frames': 50_000,
                                         'buffer_size': 100_000,
                                         'gamma': 0.97,
                                         'batch_size': 16,
                                         'solve_rewards': 15,
                                         'sync_nets': 1000,
                                         'seed': 155,
                                         'n_envs': 4,
                                         'frame_stack': 4,
                                         'steps': 4,
                                         'max_steps': None,
                                         'min_lr':5e-5,
                                         }),


          }
