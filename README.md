# Banana Navigation using Deep Q-Network (DQN) and Double DQN

CITATION: This project is derived from Udacity's Deep Reinforement Learning Project - Navigation

Author: Aaron McUmber, Ph.D.

Date: 2021-08-05

## Introduction

This project demonstrates two agents, one using Deep Q-Networks (DQN)
and the second using Double Deep Q-Networks (DDQN) to navigate a
square world to collect yellow bananas and avoid blue bananas
within a Unity ml-agent Environment. [Described here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md)

The Agent is rewarded with +1 point for collecting yellow bananas
and rewarded -1 point for collecting blue bananas. The state space
has 37 dimensions and contains the agent's velocity, ray-based
perception of objects around the agent's forward field of view.
Provided this information the agent has four discrete actions are
available:

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The episodic task is considered solved when the agent scores +13
over 100 consecutive episodes


## Getting Started

* Download the supporting modules:
1. Unity ml-agents version 0.4.0 module along with it's required dependencies: 
[here](https://github.com/Unity-Technologies/ml-agents)
2. A built version of the Unity environment provided by the 
[problem statement repo](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md)
as a direct executable for the target environment.- *Note: be
sure to select the correct environment for your machine*


## Instructions

1. Open `Report.ipynb` and run the report and follow along with 
the provided code.
  * Section 1 provides an implementation of a DQN agent
  * Section 2 provides an implementation of a DDQN agent
2. Follow the training instructions within the report to train your agent
and visualize its progress over time
