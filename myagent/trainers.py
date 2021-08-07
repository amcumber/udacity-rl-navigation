import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.optim as optim

from .agents import Agent
from .environments import EnvironmentMgr


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass    
    
    @abstractmethod
    def eval(self):
        pass
    
class DQNTrainer(Trainer):
    def __init__(
        self,
        agent: Agent,
        env: EnvironmentMgr,
        n_episodes=2000,
        max_t=10000,
        eps_start=1.0,
        eps_end=0.01, 
        eps_decay=0.995,
        print_every=100,
        solved=-100,
        save_file='my-checkpoint.pth',
    ):
        """Deep Q-Learning.

        Parameters
        ----------
        agent : Agent
            agent to act upon
        # env : UnityEnvironmentMgr
        #     environment manager containing enter and exit methods to call UnityEnvironment
        env : UnityEnvironment
            Unity Environment - DO NOT CLOSE in v0.4.0 - this will cause you to be locked 
            out of your environment... NOTE TO UDACITY STAFF - fix this issue by upgrading
            UnityEnvironemnt requirements. See 
            https://github.com/Unity-Technologies/ml-agents/issues/1167
        n_episodes (int): 
            maximum number of training episodes
        max_t (int): 
            maximum number of timesteps per episode
        eps_start (float): 
            starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): 
            minimum value of epsilon
        eps_decay (float):
            multiplicative factor (per episode) for decreasing epsilon
        print_every : int
            update terminal with information for every specified iteration, [100]
        solved : float
            score to be considered solved
        save_file: str
            file to save network weights
        """
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_t = max_t 
        self.eps_start = eps_start
        self.eps_end = eps_end 
        self.eps_decay = eps_decay
        
        self.solved = solved # TODO
        self.print_every = print_every
        self.save_file = save_file
        
        self.scores_ = None
    
    def train(self):
        self.env.start()
        scores = []                        # list containing scores from each episode
        scores_window = self.init_scores_window()
        eps = self.init_eps()
        score = 0
        try:
            for i_episode in range(self.n_episodes):
                state = self.env.reset()
                score = 0
                for t in range(self.max_t):
                    action = self.agent.act(state, eps)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break 
                scores_window.append(score)       # save most recent score
                scores.append(score)              # save most recent score
                self.scores_ = scores
                eps = self.update_eps(eps)
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(
                        i_episode+1, np.mean(scores_window)
                    ), end=""
                )
                if (i_episode + 1) % self.print_every == 0:
                    print(
                        '\rEpisode {}\tAverage Score: {:.2f}'.format(
                            i_episode+1, np.mean(scores_window))
                    )
                if np.mean(scores_window)>=self.solved:
                    print(
                        '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        (i_episode + 1), np.mean(scores_window))
                    )
                    self.agent.save(self.save_file)
                    break
            self.scores_ = scores  # this may be unecessary
            return scores
        except KeyboardInterrupt:
            print(f'Halted Early! @{i_episode+1} w/ Avg Score: {np.mean(scores_window):.2f}')
            return scores
        
    def eval(self, n_episodes=3, t_max=1000):
        ## scores_window 
        scores = []
        for i in range(n_episodes):
            state = self.env.reset()
            score = 0
            for j in range(t_max):
                action = self.agent.act(state)
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    break
            scores.append(score)
            self.scores_ = scores
            print(
                f'\rEpisode {i+1}\tFinal Score {np.mean(scores):.2f}', 
                end=""
            )
        return scores


    def init_eps(self):
        return self.eps_start
    
    def init_scores_window(self, l=100):
        return deque(maxlen=l)
    
    def update_eps(self, eps):
        return max(self.eps_end, self.eps_decay * eps)
