# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
from torch.distributions import Normal


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class PPOAgent(Agent):
    def __init__(self, n_actions: int, actor, device):
        super(PPOAgent, self).__init__(n_actions)
        self.device = device
        self.actor = actor.to(self.device)

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        with torch.no_grad():
            mu, var = self.actor(state)

            dist = Normal(mu, torch.sqrt(var))
            a = dist.sample()
            a = torch.clamp(a, -1.0, 1.0)
        return a.cpu().numpy()
    