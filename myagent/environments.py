import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod

import gym

class EnvironmentNotLoadedError(Exception):
    pass


class EnvironmentMgr(ABC):
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, e_type, e_value, e_traceback):
        pass
    
    @abstractmethod
    def step(self, action)-> '(next_state, reward, done, env_info)':
        pass
    
    @abstractmethod
    def reset(self) -> 'state':
        pass
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def get_env(stream):
        pass
    

class GymEnvironmentMgr(EnvironmentMgr):
    def __init__(self, scenario, seed=42):
        self.scenario = scenario
        self.seed = seed
        
        self.env = None
        self.action_size = None
        self.state_size = None
        
    def __enter__(self):
        return self.start()
    
    def __exit__(self, e_type, e_value, e_traceback):
        pass
    
    def step(self, action)-> '(next_state, reward, done, env_info)':
        return self.env.step(action)
    
    def reset(self) -> 'state':
        if self.env is None:
            raise EnvironmentNotLoadedError('Environment Not Initialized, run start method')
        return self.env.reset()
    
    def start(self):
        if self.env is None:
            self.env = self.get_env(self.scenario)
            self.env.seed(self.seed)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.n
        return self.env
    
    @staticmethod
    def get_env(scenario):
        return gym.make(scenario)
    