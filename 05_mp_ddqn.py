"""
        *********  Using Multipocessing ***********
Solving Lunar Lander using Deep-Q Learning Method
Network: Duel Network with one hidden layer and two heads (Value and Action_Advantage)
Loss : Bellman equation loss= Reward + Gamma x Q(s,a) vs Max(Q(s`,a)) using the target network

Results
-------
I was able to solve this in 9 minutes (dead), which is 2 minutes faster than the exact
same configurations without the multiprocess.

Another run using N_MP as multiplier (i.e. increase batch size):
    Solved in 0:07:45
    Batch_size = 128 x 4 # BATCH_MULT = 4
    LearningRate= 1e-3
    N Environments = 1
    Hidden Size = 348

Notes:
------
* I didn't notice any improvement when increasing the number of mp processes. Seems like
one is enough for LunarLander at least
* Increaseing number of taken stesp improves time but the agent becomes a bit clumzy!
"""
import torch
import torch.multiprocessing as mp

import os
# import gym
import ptan
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from lib import mp_utils, model, data, utils
from collections import deque

THREADS = 4


@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = str(THREADS)
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', default=False,
                        help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help='Save a copy of the trained network in current directory as "game_dqn.dat"')
    parser.add_argument('--env', default='invaders',
                        help='name of the game: invaders(default), pong, breakout')
    parser.add_argument('--cuda', action='store_true',
                        help='Train on GPU when available')
    args = parser.parse_args()

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    frames = mp.Value('i', 0)
    episodes = mp.Value('i', 0)

    params = data.params[args.env]

    env = utils.createEnvs(params)[0]
    shape = env.observation_space.shape
    actions = env.action_space.n

    net = model.DDQN(shape, actions).to(device)
    net.share_memory()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    buffer = ptan.experience.ExperienceReplayBuffer(None, params.buffer_size)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    exp_queue = mp.Queue(THREADS)
    proc_list = []
    for n in range(THREADS):
        proc = mp.Process(target=mp_utils.data_fun_global, name=str(
            n), args=(net, exp_queue, params, frames, episodes, device))
        proc.start()
        proc_list.append(proc)

    generator = mp_utils.MPBatchGenerator(buffer, exp_queue, params.init_replay,
                                          params.batch_size, 1)
    pt = time()
    loss = 0.0
    start_time = datetime.now()
    total_rewards = deque(maxlen=100)
    # epoch = 0
    frame = 0
    for batch in generator:
        new_reward = generator.pop_rewards_idx_eps()
        if new_reward:
            total_rewards.extend(new_reward)
            mean = np.mean(total_rewards)
            if mean > params.solve_rewards:
                duration = timedelta(
                    seconds=(datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save:
                    file_name = env.game + '_dqn_mp.dat'
                    torch.save(net.state_dict(), file_name)
                if args.play:
                    play(env, agent)
                break
            if time()-pt > 1:
                fps = int((frames.value - frame)/(time()-pt))
                frame = frames.value
                episode = episodes.value
                print(
                    f'{frame:6,} done:{episode:6,} reward:{new_reward[0]:7.2f},mean:{mean:7.2f}, loss:{loss:6.3f}, speed:{fps:5} fps, epsilon:{generator.epsilon:4.2f}')
                pt = time()

        optimizer.zero_grad()
        loss = utils.calc_loss_dqn(
            batch, net, tgt_net, params.gamma**params.steps, device=device)
        loss.backward()
        optimizer.step()
        # epoch += 1
        if generator.frame % params.sync_nets == 0:
            tgt_net.sync()
        del batch

    exp_queue.close()
    exp_queue.join_thread()

    for p in proc_list:
        p.kill()
        p.join()
