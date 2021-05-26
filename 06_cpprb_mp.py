#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:26:12 2021

@author: ayman
"""
# %% prerun
# from multiprocessing import Process, Event, SimpleQueue

import ptan
import torch
import torch.multiprocessing as mp

import numpy as np
from collections import namedtuple
from datetime import datetime
from cpprb import MPReplayBuffer, create_env_dict, ReplayBuffer


from lib import model, utils, data


Observation = namedtuple(
    'Observation', ['rew', 'done', 'obs', 'next_obs', 'act'])


def createEnv(env_id):
    env = utils.atari_wrappers.make_atari(
        env_id, skip_maxskip=True, skip_noop=True)
    return utils.atari_wrappers.wrap_deepmind(env)


def preprocess(obs):
    obs = np.expand_dims(obs, 0)
    return torch.FloatTensor(obs)


def data_fun(params, agent, exp_queue, done_training, frames, episodes):
    env = createEnv(params.env)
    obs = env.reset()
    while not done_training.is_set():
        frames.value += 1
        action = agent(obs)[0].item()
        next_obs, rew, done, _ = env.step(action)
        if done:
            episodes.value += 1
            exp_queue.put({'rew': rew, 'done': done, 'obs': obs,
                           'next_obs': obs, 'act': action})
            obs = env.reset()
        else:
            exp_queue.put({'rew': rew, 'done': done, 'obs': obs,
                           'next_obs': next_obs, 'act': action})
            obs = next_obs
    pass


def calc_loss_dqn(batch, net, tgt_net, gamma, device, priority=False):
    """Return batch loss and priority updates (priority must be True) """

    if priority:
        rewards, dones, states, last_states, actions, weights, indexes = batch.values()
    else:
        rewards, dones, states, last_states, actions = batch.values()

    states_v = torch.FloatTensor(states).to(device)
    rewards_v = torch.FloatTensor(rewards).squeeze(-1).to(device)

    q_s_a = net(states_v)[range(len(actions)), actions.squeeze(-1)]
    with torch.no_grad():
        last_states_v = torch.FloatTensor(last_states).to(device)
        best_next_q_v = tgt_net.target_model(last_states_v).max(dim=1)[0]
        best_next_q_v[dones.squeeze(-1)] = 0.0
        exp_q_s_a = rewards_v + gamma * best_next_q_v.detach()
    if priority:
        weights_v = torch.FloatTensor(weights).to(device)
        l = (q_s_a - exp_q_s_a)**2
        loss_v = l * weights_v
        return loss_v.mean(), (loss_v + 1e-5).data.cpu().numpy(), indexes
    return torch.nn.functional.mse_loss(q_s_a, exp_q_s_a)


# def learner(global_rb,queues):
#     batch_size = 64
#     n_warmup = 100
#     n_training_step = int(1e+4)
#     explorer_update_freq = 100

#     model = MyModel()

#     while global_rb.get_stored_size() < n_warmup:
#         time.sleep(1)

#     for step in tqdm(range(n_training_step)):
#         sample = global_rb.sample(batch_size)

#         model.train(sample)
#         absTD = model.abs_TD_error(sample)
#         global_rb.update_priorities(sample["indexes"],absTD)

#         if step % explorer_update_freq == 0:
#             w = model.weights
#             for q in queues:
#                 q.put(w)

# %% run_cell
if __name__ == "__main__":
    ENV_ID = 'pong'
    THREADS = 4
    mp.set_start_method('spawn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    params = data.params[ENV_ID]
    test_env = createEnv(params.env)

    shape = test_env.observation_space.shape
    actions = test_env.action_space.n
    env_dict = create_env_dict(test_env)

    net = model.DDQN(shape, actions).to(device)
    net.share_memory()
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(
        net, selector, device=device, preprocessor=preprocess)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    done_training = mp.Event()
    done_training.clear()

    global_rb = ReplayBuffer(size=10_000, env_dict=env_dict)
    exp_queue = mp.Queue(maxsize=1)

    frames = mp.Value('i', 0)
    episodes = mp.Value('i', 0)

    procs = [mp.Process(target=data_fun,
                        args=(params, agent, exp_queue, done_training, frames, episodes)) for _ in range(THREADS)]

    for p in procs:
        p.start()

    total_rewards = []
    start = datetime.now()
    mean = -float('inf')

    while not done_training.is_set():
        while not exp_queue.empty():
            state = exp_queue.get()
            total_rewards.append(state['rew'])
            global_rb.add(**state)
            del state
        if global_rb.get_stored_size() < params.init_replay:
            continue
        if (datetime.now()-start).seconds > 3:
            mean = np.mean(total_rewards[-100:])
            print(f'{frames.value:7,} done: {episodes.value:5} mean: {mean:.3f}')
            start = datetime.now()

        batch = global_rb.sample(params.batch_size)
        optimizer.zero_grad()
        loss = calc_loss_dqn(batch, net, tgt_net, params.gamma, device, False)
        loss.backward()
        optimizer.step()
        del batch
        if frames.value % params.sync_nets == 0:
            tgt_net.sync()

        if mean > 10:
            done_training.set()

    exp_queue.close()
    exp_queue.join_thread()

    for p in procs:
        p.terminate()
        p.join()
