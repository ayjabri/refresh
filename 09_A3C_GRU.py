#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempting to solve Atari games using A2C

@author: Ayman Jabri
"""
import os
import torch
import ptan
from lib import data, utils, mp_utils, model
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp


ALGORITHM = 'A3C_GRU'
FORKS = 4
MINI_BATCH = 16


if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    message = '*'*10 + '  A3C GRU on Atari ' +'*'*10
    args = utils.argpars_dqn(message)
    params = data.params[args.env]
    utils.update_params(params, args)
    
    # For A2C/A3C to converge we need a lot of environments to draw observations from
    # This will ensure sample i.i.d (somehow!)
    params.n_envs = max(params.n_envs, 8)

    device = 'cuda' if args.cuda else 'cpu'

    env = utils.createEnvs(params, stack_frames=params.frame_stack)[0] 
    
    shape = env.observation_space.shape
    actions = env.action_space.n
    net = model.A2Cgru(shape, actions)
    net.to(device)
    net.share_memory()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    
    mean_monitor = utils.MeanRewardsMonitor(env, net, ALGORITHM, params.solve_rewards)

    writer = SummaryWriter(logdir=mean_monitor.runs_dir,comment=params.frame_stack)
  
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20000,
                                                            cooldown=20000, verbose=True, min_lr=params.min_lr)

    
    print('# Parameters: ', utils.count_parameters(net))
    print(net)
    print('*'*10, ' Start Training ',
          env.game, ' {} '.format(device), '*'*10)    
    
    exp_queue = mp.Queue(maxsize=FORKS)
    procs = []
    for fork in range(FORKS):    
        p = mp.Process(target= mp_utils.a3c_data_fun, args= (net, exp_queue, params, MINI_BATCH, device))
        p.start()
        procs.append(p)
    
    batch = []
    frame = 0
    BATCH_SIZE = FORKS * MINI_BATCH
    mean = None
    try:
        with ptan.common.utils.RewardTracker(writer) as tracker:
            while True:
                data = exp_queue.get()
                if isinstance(data, mp_utils.EpisodeEnd):
                    frame += data.step
                    mean = tracker.reward(data.reward, frame)
                    if mean_monitor(mean):
                        break
                else:
                    batch.extend(data)
                
                if len(batch) < 256:
                    continue
                
                #### Training ###
                optimizer.zero_grad()
                value_loss, policy_loss, entropy_loss = utils.calc_loss_a2c(batch, net, params, device=device)
                total_loss = value_loss + policy_loss + entropy_loss
                total_loss.backward()
                optimizer.step()
                
                if mean:
                    lr_scheduler.step(mean_monitor.best_reward)
                    writer.add_scalar('LearningRate', scalar_value=lr_scheduler._last_lr, global_step=frame)
                batch.clear()
    
    finally:
        exp_queue.close()
        exp_queue.join_thread()
        for p in procs:
            p.terminate()
            p.join()
