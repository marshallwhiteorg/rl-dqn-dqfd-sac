import copy
import math
import os
from collections import namedtuple

import gym
import ipywidgets as widgets
import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import demonstrator
import pickle
import math
import DQfD



###train demonstrator
env = gym.make('LunarLander-v2')

dqfd = demonstrator.DQN(8, 4)

dqfd_target = demonstrator.DQN(8, 4)

num_steps = 500000
num_saves = 5
replay_size = 50000
prepopulate_steps = 50000
state_size = 8

gamma = 0.99

optimizer = torch.optim.Adam(dqfd.parameters(), lr = 0.001, weight_decay = 0.00001)

memory = demonstrator.ReplayMemory(replay_size, state_size, batch_size=64)
memory.populate(env, prepopulate_steps)
memory.set_batch_size(64)

rewards = []
returns = []
lengths = []
losses = []

t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)

i_episode = 0
t_episode = 0

state = env.reset()

eps = 0.01

exploration = DQfD.ExponentialSchedule(1.0, 0.05, 500000)

final_model = {}

for t_total in range(num_steps):
    if t_total % 100 == 0:
        print("Step: ", t_total)

    if t_total % 50000 == 0:
        for i in range(10):
            demonstrator.render(env, dqfd)

    eps = exploration.value(t_total)

    if (np.random.rand(1)[0] > eps):

        _, A = torch.max(dqfd.forward(torch.as_tensor(state, dtype=torch.float)), 0)
        A = A.item()

    else:

        A = np.random.randint(0, env.action_space.n, 1)[0]

    S_next, reward, done, info = env.step(A)

    rewards.append(reward)

    memory.add(state, A, reward, S_next, done)

    state = S_next

    if t_total % 4 == 0:

        batch, indices = memory.sample()

        loss, td_errors = DQfD.train_dqn_batch_demonstrator(optimizer, batch, dqfd, dqfd_target, gamma, expert=False, DQN=True)

        losses.append(loss)

    if t_total % 10000 == 0:
        dqfd_target.load_state_dict(dqfd.state_dict())

    if done:

        G = 0

        for i, r in enumerate(rewards):
            if i == len(rewards) - 1:
                G += r
            else:
                G = gamma * (G + r)


        lengths.append(t_episode)
        returns.append(G)
        rewards = []

        i_episode += 1
        t_episode = 0

        state = env.reset()

    else:

        t_episode += 1

final_model['100_0'] = copy.deepcopy(dqfd)

checkpoint2 = {key: dqfd.custom_dump() for key, dqfd in final_model.items()}
torch.save(checkpoint2, f'checkpoint_final_DQN.pt')


returns = np.array(returns)
lengths = np.array(lengths)
losses = np.array(losses)
pickle.dump(returns, open("returns_DQN.p", "wb"))
pickle.dump(lengths, open("lengths_DQN.p", "wb"))
pickle.dump(losses, open("losses_DQN.p", "wb"))
plt.plot(np.arange(0, len(returns)), returns)
plt.plot(np.arange(0, len(returns)), DQfD.rolling_average(returns, window_size=round(len(returns)*0.1)))
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.show()

plt.plot(np.arange(0, len(lengths)), lengths)
plt.plot(np.arange(0, len(lengths)), DQfD.rolling_average(lengths, window_size=round(len(lengths)*0.1)))
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.show()

plt.plot(np.arange(0, len(losses)), losses)
plt.plot(np.arange(0, len(losses)), DQfD.rolling_average(losses, window_size=round(len(losses)*0.1)))
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()

for i in range(10):
    demonstrator.render(env, dqfd)




###loads demonstration data for supervised learning from demonstrator


env = gym.make('LunarLander-v2')

checkpoint = torch.load('checkpoint_LunarLander.pt')

for key, value in checkpoint.items():
    dqn = demonstrator.DQN.custom_load(value)

demonstration_replay = demonstrator.ReplayMemory(200000, 8, per_a=1, batch_size=64)

demonstration_replay.add_demonstrations(env, dqn)

pickle.dump(demonstration_replay, open("demonstration_replay.p", "wb"))







###Train agent from expert using supervised learning


env = gym.make('LunarLander-v2')

dqfd = DQfD.DQFD(8, 4)

dqfd_target = DQfD.DQFD(8, 4)

gamma = 0.99

optimizer = torch.optim.Adam(dqfd.parameters(), lr = 0.001, weight_decay = 0.00001)

new_demonstration_buffer = pickle.load(open("demonstration_replay.p", "rb"))

new_demonstration_buffer.set_batch_size(64)

demonstrator_model = {}

for i in range(5000):

    batch, indices = new_demonstration_buffer.sample()

    loss, td_errors, loss1, loss2, loss3 = DQfD.train_dqn_batch_demonstrator(optimizer, batch, dqfd, dqfd_target, gamma, batch_size=64)
    new_demonstration_buffer.set_priorities(indices, td_errors)

    if i % 100 == 0:
        dqfd_target.load_state_dict(dqfd.state_dict())

    if i % 100 == 0:
        print(i)
        print(loss)
        print(loss3)
    if i % 1000 == 0 and i != 0:
        for i in range(10):
            demonstrator.render(env, dqfd)

supervised_model = {}

supervised_model['1'] = copy.deepcopy(dqfd)

checkpoint = {key: dqfd.custom_dump() for key, dqfd in supervised_model.items()}
torch.save(checkpoint, 'supervised_model.pt')















##Train agent using reinforcement learning methods





env = gym.make('LunarLander-v2')

checkpoint = torch.load('supervised_model.pt')

for key, value in checkpoint.items():
    dqfd = DQfD.DQFD.custom_load(value)

for key, value in checkpoint.items():
    dqfd_target = DQfD.DQFD.custom_load(value)

num_steps = 500000
num_saves = 5
replay_size = 50000
prepopulate_steps = 50000
state_size = 8

gamma = 0.99

optimizer = torch.optim.Adam(dqfd.parameters(), lr = 0.001, weight_decay = 0.00001)

memory = demonstrator.ReplayMemory(replay_size, state_size, batch_size=64)
memory.populate(env, prepopulate_steps)
memory.set_batch_size(64)

rewards = []
returns = []
lengths = []
losses = []

t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)

i_episode = 0
t_episode = 0

state = env.reset()

eps = 0.01

exploration = DQfD.ExponentialSchedule(1.0, 0.05, 500000)

final_model = {}

for t_total in range(num_steps):
    if t_total % 100 == 0:
        print("Step: ", t_total)

    if t_total % 50000 == 0:
        for i in range(10):
            demonstrator.render(env, dqfd)

    eps = exploration.value(t_total)

    if (np.random.rand(1)[0] > eps):

        _, A = torch.max(dqfd.forward(torch.as_tensor(state, dtype=torch.float)), 0)
        A = A.item()

    else:

        A = np.random.randint(0, env.action_space.n, 1)[0]

    S_next, reward, done, info = env.step(A)

    rewards.append(reward)

    memory.add(state, A, reward, S_next, done)

    state = S_next

    if t_total % 4 == 0:

        batch, indices = memory.sample()

        loss, td_errors, loss1, loss2 = DQfD.train_dqn_batch_demonstrator(optimizer, batch, dqfd, dqfd_target, gamma, expert=False)

        losses.append(loss)

    if t_total % 10000 == 0:
        dqfd_target.load_state_dict(dqfd.state_dict())

    if done:

        G = 0

        for i, r in enumerate(rewards):
            if i == len(rewards) - 1:
                G += r
            else:
                G = gamma * (G + r)


        lengths.append(t_episode)
        returns.append(G)
        rewards = []

        i_episode += 1
        t_episode = 0

        state = env.reset()

    else:

        t_episode += 1

final_model['final_model'] = copy.deepcopy(dqfd)

checkpoint2 = {key: dqfd.custom_dump() for key, dqfd in final_model.items()}
torch.save(checkpoint2, 'final_model.pt')


returns = np.array(returns)
lengths = np.array(lengths)
losses = np.array(losses)
pickle.dump(returns, open("returns.p", "wb"))
pickle.dump(lengths, open("lengths.p", "wb"))
pickle.dump(losses, open("losses.p", "wb"))
plt.plot(np.arange(0, len(returns)), returns)
plt.plot(np.arange(0, len(returns)), DQfD.rolling_average(returns, window_size=round(len(returns)*0.1)))
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.show()

plt.plot(np.arange(0, len(lengths)), lengths)
plt.plot(np.arange(0, len(lengths)), DQfD.rolling_average(lengths, window_size=round(len(lengths)*0.1)))
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.show()

plt.plot(np.arange(0, len(losses)), losses)
plt.plot(np.arange(0, len(losses)), DQfD.rolling_average(losses, window_size=round(len(losses)*0.1)))
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()

for i in range(10):
    demonstrator.render(env, dqfd)
