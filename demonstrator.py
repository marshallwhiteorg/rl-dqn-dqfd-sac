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
import math



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.model = torch.nn.Sequential()
        self.model.add_module("linear 1", torch.nn.Linear(self.state_dim, self.hidden_dim))
        self.model.add_module("relu 1", torch.nn.ReLU())

        for i in range(1, num_layers - 2):
            self.model.add_module("linear " + str((i+1)), torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            self.model.add_module("relu " + str((i+1)), torch.nn.ReLU())

        self.model.add_module("output", torch.nn.Linear(self.hidden_dim, self.action_dim))



    def forward(self, states) -> torch.Tensor:

        return self.model.forward(states)

    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }




Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones', 'n_rewards', 'n_states')
)





def render(env, dqfd):

    state = env.reset()
    env.render()

    while True:
        action = dqfd.forward(torch.tensor(state, dtype=torch.float)).argmax().item()
        state, _, done, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()



class ReplayMemory:
    def __init__(self, max_size, state_size, n=10, gamma=0.99, batch_size=64, per_exp=0.4, per_a=0.001):

        self.max_size = max_size
        self.state_size = state_size

        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)

        self.per_a = per_a
        self.per_exp = per_exp

        self.priorities = torch.zeros((max_size, 1)) + self.per_a

        self.idx = 0

        self.size = 0

        self.n = n
        self.gamma = gamma
        self.batch_size = batch_size



    def add(self, state, action, reward, next_state, done):

        self.states[self.idx] = torch.as_tensor(state, dtype = torch.double)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = torch.as_tensor(next_state, dtype=torch.double)
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size

        self.size = min(self.size + 1, self.max_size)


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.n_rewards = torch.empty((batch_size, 1))
        self.n_states = torch.empty((batch_size, self.state_size))



    def add_demonstrations(self, env, dqn):

        S = env.reset()

        for i in range(self.max_size):

            if i % 1000 == 0:
                print(i)

            A = dqn(torch.tensor(S, dtype=torch.float)).argmax().item()

            S_next, reward, done, info = env.step(A)

            self.add(S, A, reward, S_next, done)

            if done:

                S = env.reset()

            else:

                S = S_next


    def calculate_n_rewards(self, sample_indices):

        for i, idx in enumerate(sample_indices):

            discounted_reward = 0

            if idx + self.n < len(self.states):

                self.n_states[i] = (self.states[idx + self.n])

            else:

                self.n_states[i] = torch.as_tensor([0, 0, 0, 0, 0, 0, 0, 0])

            for j in range(self.n):

                if idx + j < self.rewards.shape[0]:

                    discounted_reward += self.gamma * self.rewards[idx + j]

                else:

                    break

            self.n_rewards[i] = discounted_reward





    def sample(self) -> Batch:

        if self.batch_size >= self.size:
            batch = Batch(self.states, self.actions, self.rewards, self.next_states, self.dones)
            return batch

        probabilities = self.priorities.numpy()

        probabilities = np.power(probabilities, self.per_exp)

        probabilities = probabilities / np.sum(probabilities)

        probabilities = np.reshape(probabilities, probabilities.shape[0])

        sample_indices = np.random.choice(self.size, self.batch_size, p=probabilities)

        self.calculate_n_rewards(sample_indices)

        batch = Batch(self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices], self.next_states[sample_indices], self.dones[sample_indices], self.n_rewards, self.n_states)

        return batch, sample_indices



    def set_priorities(self, sample_indices, td_errors):
        td_errors = td_errors.detach().numpy()
        for i, idx in enumerate(sample_indices):
            self.priorities[idx] = abs(td_errors[i][0]) + self.per_a




    def populate(self, env, num_steps):

        S = env.reset()

        for i in range(num_steps):

            A = np.random.randint(0, env.action_space.n, 1)[0]

            S_next, reward, done, info = env.step(A)

            self.add(S, A, reward, S_next, done)

            if done:

                S = env.reset()

            else:

                S = S_next
