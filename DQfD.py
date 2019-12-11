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

Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones', 'n_rewards', 'n_states')
)

class DQFD(nn.Module):

    def __init__(self, state_dim, action_dim, num_layers=3, hidden_dim=256):

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






def train_dqn_batch_demonstrator(optimizer, batch, dqn_model, dqn_target, gamma, batch_size=64, expert=True, DQN=False) -> float:
    JDQ_predictions = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=True)
    JDQ_targets = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=False)
    JN_predictions = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=True)
    JN_targets = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=False)
    JE_predictions = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=True)
    JE_targets = torch.zeros([batch_size, 1], dtype=torch.float, requires_grad=True)

    JDQ_predictions = dqn_model.forward(batch.states)
    JDQ_predictions = torch.gather(JDQ_predictions, 1, batch.actions)

    for i in range(batch_size):
        if batch.dones[i]:
            JDQ_targets[i] = batch.rewards[i]
        else:
            JDQ_targets[i] = torch.add(batch.rewards[i], gamma * dqn_target.forward(batch.next_states[i]).max())

    JDQ_targets = JDQ_targets.clone().detach()
    assert JDQ_predictions.requires_grad, 'values tensor should require gradients'
    assert (
        not JDQ_targets.requires_grad
    ), 'target_values tensor should require gradients'

    JN_predictions = dqn_model.forward(batch.states)
    JN_predictions = torch.gather(JN_predictions, 1, batch.actions)

    if (DQN == True):
        loss = F.mse_loss(JDQ_predictions, JDQ_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), JDQ_targets - JDQ_predictions

    for i in range(batch_size):
        if batch.n_states[i][0] == 10000000000:
            JN_targets[i] = batch.n_rewards[i]
        else:
            JN_targets[i] = torch.add(batch.n_rewards[i], math.pow(gamma, 10) * dqn_target.forward(batch.n_states[i]).max())

    JN_targets = JN_targets.clone().detach()
    assert JN_predictions.requires_grad, 'values tensor should require gradients'
    assert (
        not JN_targets.requires_grad
    ), 'target_values tensor should require gradients'

    if (expert == False):
        loss1 = F.mse_loss(JDQ_predictions, JDQ_targets)
        loss2 = F.mse_loss(JN_predictions, JN_targets)

        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), JDQ_targets - JDQ_predictions, loss1.item(), loss2.item()


    Q_values_predictions = dqn_model.forward(batch.states)
    Q_values_temp = dqn_model.forward(batch.states)

    print("test")

    JE_predictions = torch.gather(Q_values_predictions, 1, batch.actions)

    for i in range(batch_size):
        temp = batch.actions[i]
        for j in range(Q_values_temp.shape[1]):
            if temp != j:
                Q_values_temp[i][j] += 0.8

    JE_targets = Q_values_temp.max(dim=1)[0]

    JE_targets = JE_targets.clone().detach()
    JE_targets = JE_targets.view(-1, 1)

    assert JE_predictions.requires_grad, 'values tensor should require gradients'
    assert (
        not JE_targets.requires_grad
    ), 'target_values tensor should require gradients'

    loss1 = F.mse_loss(JDQ_predictions, JDQ_targets)
    loss2 = F.mse_loss(JN_predictions, JN_targets)
    loss3 = F.mse_loss(JE_predictions, JE_targets)

    loss = loss1 + loss2 + loss3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), JDQ_targets - JDQ_predictions, loss1.item(), loss2.item(), loss3.item()






def rolling_average(data, *, window_size):

    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]



class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):

        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        self.a = self.value_from
        self.b = math.log(self.value_to / self.value_from) / (self.num_steps - 1)

    def value(self, step) -> float:

        if step <= 0:
            return self.value_from

        elif step >= self.num_steps - 1:
            return self.value_to

        value = self.a * math.exp(self.b * step)

        return value
