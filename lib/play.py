#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:41:20 2021

@author: ayman
"""

import torch
import time
import numpy as np

def play(env, agent=None, device='cpu', wait=0.0):
    state = env.reset()
    rewards = 0
    while True:
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
    
        
    