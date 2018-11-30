[//]: # (Image References)
[image1]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p3_collab-compet/tennis.gif "Trained Agent"
[image2]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p3_collab-compet/reward_plot.png?raw=true"Reward "Plot"



# Project 3: Collaborate and Competition

## Introduction

For this project, I have worked with [Tennis](ttps://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment 

### Trained Agents
![Trained Agent][image1]


In this environment,  two agents control rackets to bounce a ball over a net. If agent hits the ball over the net, it receives a rewawd +0.1. If an agent lets a ball hit the ground or hits the ball out of the bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.


The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are availabel, corresponding to movement toward (or away from) the net, and jumping.


The task is episodic, and in order to solve the environment,agent must get an average score of +0.5 (over 100 consecutive episodes, after taking the **maximum** over both agents). Specifically,

- After each episode, we add up the rewards that agent received (with discounting). This yields 2 (potentially different) score. We then take the maximum of these 2 scores.
- This yields a single **score** for each epsiode.

The environment is considered solved, when the average (over 100 episodes) of those **score** is at least +0.5.


### Learning Algorithm

---

The algorithm chosen to solve this environment is Mulit-Agent Deep Determinstic Policy Gradient (MADDPG). MADDPG is a multi agent version of DDPG. DDPG is well suited to continuous control task and this extend to multi-agent RL.

In MADDPG, each agent has its own Actor and Critic. Agents share a common experience replay buffer which contains tuples with state and actions for all agents. But critic of each agent have access to state and action of its actor and all other actor in the environment too.

For this environment I created two separate DDPG agents. Each actor takes a 24 dimensional state input. Each critic takes a concatenation of the state (48 dimensions) and actions (4 dimensions) from both the agents.

### Hyperparameters
* Two hidden layers with 128 node, 150 nodes respectively.
* Relu activation function for all non linear activation.
* Batch size : 128
* Tau(soft update): 1e-5
* discount rate: 0.99
* learning rate for critc and actor : 0.0001

### Plot of Rewards

---
![Reward Plot][image2]


### Ideas for Future
---
* It would be interesting to apply this algorithm to a more complex environment, like the 2 on 2 SoccerTwos as this would involve agents working in both a collaborative and competitive manner simultaneously. Given that each agent is able to learn it's own reward function, this should be feasible.
* Will try to implement Multi agent PPO algorithm to the 2 on 2 soccerTwos.