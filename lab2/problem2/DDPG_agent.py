# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
from abc import ABC, abstractmethod

class Noise(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class LowPassNoise(Noise):
    def __init__(self, n_actions: int, device, sigma : float = 0.2, mu: float = 0.15):
        self.last_noise_signal = torch.zeros(n_actions, device=device)
        self.mu = mu
        self.sigma = sigma
        self.n_actions = n_actions
        self.device = device
    
    def sample(self):
        w = self.sigma * torch.randn(self.n_actions, device = self.device)
        n_t = -self.mu * self.last_noise_signal + w
        self.last_noise_signal = n_t
        return n_t
    
    def reset(self):
        self.last_noise_signal = torch.zeros(self.n_actions, device=self.device)

    def set_sigma(self, sigma):
        self.sigma = sigma

class UncorrNormalNoise(Noise):
    def __init__(self, n_actions: int, device, sigma: float = 0.2):
        self.sigma = sigma
        self.device = device
        self.n_actions = n_actions
    
    def sample(self):
        noise = self.sigma * torch.randn(self.n_actions, device=self.device)
        return noise
    
    def reset(self):
        pass


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
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class DDPGAgent(Agent):
    def __init__(self, n_actions: int, actor, device, noise: Noise):
        super(DDPGAgent, self).__init__(n_actions)
        self.noise = noise
        self.device = device
        self.actor = actor.to(self.device)

    def forward(self, state, explore = True):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)

        state = state.to(self.device)

        with torch.no_grad():
            a = self.actor(state)

            if explore:
                a += self.noise.sample()
            a = torch.clamp(a, -1.0, 1.0)
        return a.cpu().numpy()
    


