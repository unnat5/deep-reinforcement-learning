[//]: # (Image References)

[image1]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p3_collab-compet/tennis.gif "Trained Agent"
[image2]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p3_collab-compet/reward_plot.png?raw=true "Reward Plot"


# Project 3: Collaboration and Competition

### Introduction

For this project, I have worked with [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Learning Algorithm
---

The algorithm chosen to solve this environment is Multi-Agent Deep Deterministic Policy Gradient (MADDDPG). MADDPG is a multi agent version of DDPG.DDPG is well suited to continuous control task and this extends to multi-agent RL.

In MADDPG, each agent has its own Actor and Critic. Agents share a common experience replay buffer which contains tuples with states and actions from all agents. But critic of each agent have access to state of its actor and all other actor in the environment.

For this environment I created two separate DDPG agents. Each actor takes a 24 dimensional state input. Each critic takes a concatenation of the states (48 dimensions) and actions (4 dimensions) from both agents

### Hyperparameters 
* Two hidden_layers with 128,150 nodes respectively.
* Relu activation function for all non linear activation.
* Batch_size = 128
* Tau(soft_update) = 1e-5
* discount_factor = 0.99
* learning_rate for critic and actor = 0.0001

### Plot of Rewards
---

![Reward Plot][image2]

### Ideas for Future
---
* It would be interesting to apply this algorithm to a more complex environment, like the 2 on 2 SoccerTwos as this would involve agents working in both a collaborative and competitive manner simultaneously. Given that each agent is able to learn it's own reward function, this should be feasible.

