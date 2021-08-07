## ~~! From Deep Q Network Exercise from Udacity's Deep Reinforement Learning Course
## DDQN CITATION:  Z Wang, et al. Dueling Network Architectures for Deep
#                  Reinforcement Learning. arXiv, 5 Apr 2016, 1511.06581v3
#                  (https://arxiv.org/pdf/1511.06581.pdf)
import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .qnetworks import QNetwork
from .replay_buffers import ReplayBuffer

# @dataclass
class Agent(ABC):
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass
        
    @abstractmethod
    def act(self, state, eps=0.):
        pass
    
    @abstractmethod
    def save(self, file):
        pass
    
    @abstractmethod
    def load(self, file):
        pass

        
class DQNAgent(Agent):
    """
    Deep Q-Learning Agent. Interacts with and learns from the environment.
    
    Parameters
    ----------
    state_size : int
        dimension of each state
    action_size : int
        dimension of each action
    
    Optional Parameters
    ===================
    device : str
        'cpu' | 'cuda:0'
    seed : int
        random seed
    lr : float
        learning rate
    update_every : int
        Update Q-table after number of times
    tau : float
        interpolation parameter - from Q-Network
    gamma : float
        discount for future reward
    batch_size : int
        number of instances to include within a batch
    buffer_size : int
        tbr
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str = 'cpu',
        seed_state: int = 42,
        lr: float = 5e-4,
        update_every: int = 4,
        tau: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size=int(1e5),
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.seed_state = seed_state
        self.lr = lr
        self.update_every = update_every
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.seed = random.seed(self.seed_state)
        
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed_state).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed_state).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed_state, self.device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Advance Agent"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        # experiences (Tuple[np.array]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self, file):
        """Save Q-Network"""
        torch.save(self.qnetwork_local.state_dict(), file)
            
    def load(self, file):
        """Load Q-Network"""
        self.qnetwork_local.load_state_dict(torch.load(file))
        self.qnetwork_target.load_state_dict(torch.load(file))


class DDQNAgent(Agent):
    """
    Interacts with and learns from the environment.
    
    Parameters
    ----------
    state_size : int
        dimension of each state
    action_size : int
        dimension of each action
    
    Optional Parameters
    ===================
    device : str
        'cpu' | 'cuda:0'
    seed : int
        random seed
    lr : float
        learning rate
    update_every : int
        Update Q-table after number of times
    tau : float
        interpolation parameter - from Q-Network
    gamma : float
        discount for future reward
    batch_size : int
        number of instances to include within a batch
    buffer_size : int
        tbr
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str = 'cpu',
        seed_state: int = 42,
        lr: float = 5e-4,
        update_every: int = 4,
        tau: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size=int(1e5),
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.seed_state = seed_state
        self.lr = lr
        self.update_every = update_every
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.seed = random.seed(self.seed_state)
        
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed_state).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed_state).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed_state, self.device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Advance Agent"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        # experiences (Tuple[np.array]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # CITATION ABOVE - See Wang 2016 for algorithm
        max_next_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_next_action)
      
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self, file):
        """Save Q-Network"""
        torch.save(self.qnetwork_local.state_dict(), file)
            
    def load(self, file):
        """Load Q-Network"""
        self.qnetwork_local.load_state_dict(torch.load(file))
        self.qnetwork_target.load_state_dict(torch.load(file))
