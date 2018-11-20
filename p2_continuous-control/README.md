[//]: # (Image References)

[image1]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p2_continuous-control/images/tensorboard_p2Continuous.gif "Tensor board"
[image2]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p2_continuous-control/images/smart_agent.gif "Trained Agent"
[image3]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p2_continuous-control/images/avg_agent.png "reward_agent"
[image4]: https://github.com/unnat5/deep-reinforcement-learning/blob/master/p2_continuous-control/images/avg_rolling(100).png "rolling_mean"


# Project 2: Continuous Control

## Overview
#### Environment
* __Set-up__: Double-jointed arm which can move to target locations.
* __Goal__: The agent must move it's hand  to goal location, and keep it there.
* __Agents__: The environment contain 20 agent linked to a single brain.
* __Agent Reward Function__(independent):
    *  +0.1 Each step agent's hand is in goal location.
* __Brains__: One Brain with the following observation/action space.
    * Vector Observation space : 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
    * Vector Action space (Continuous) size of 4, corresponding to torque applicable to two joints.
* __Benchmark Mean Reward__: 30

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

### Smart Agent
![Trained Agent][image2]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Algorithm used for training the Agent
* To the train the agent I have used [DDPG](https://arxiv.org/pdf/1509.02971.pdf).
* DDPG is classified as an Actor-Critc method.
* And it chooses optimal action determinstically.
* DDPG is best classified as a DQN method for continuous action spaces. 
* In **DDPG** we want to output the best believe action every time we query the action from the network.(That is determinstic policy)
    * But by adding __noise__ in the action space we can force agent to explore more.


#### Actor
* Now actor here is used to approximate **optimal policy determinstically**
* The actor is basically learning to output `argmax_aQ(s,a)` which is the best action.
* This equivalent to policy gradient method approach where we directly estimate the policy of the environment with a neural network.
* But one important thing to notice here is unlike policy gradient method where we collect trajectories and then compute cumulative future reward and then have a noisy MonteCarlo estimate(high variance) across which we calculate gradient and then optimize the network.
* In DDPG we have CRITIC which estimates optimal value function, and with help of this we compute gradinet for actor and optimize it's weight. 

#### Critic
* Learns to evaluate the optimal action value function by using **ACTOR** best believe action.

## Pipeline for DDPG
* We have four network
    1. `actor_local`
    2. `actor_target`
    3. `critic_local`
    4. `critic_target`
* First step is we collect our experience tuples which consisits of `(state, action, reward, next_state, done)`.
* We randomly initialize (xavier initialization) our network. One of the key component which I used and found very helpful in this problem was initializing local and target networks with same set of random weights.
* We first get a state from environment and then we pass it through `actor_local` and get an action, by taking this action we get next_state and reward. By this process we collect experience tuples and push it in replay buffer.
* After our replay buffer length is greater or equal to batch_size, we randomly sample from replay buffer to break sequential correlation.
* The fact we have four network is because while training our neural network we need targets, so that we can compute loss and then gradient across it. But in RL target itself is a function of weights, so we want to break this relation that is the reason we have target network which are different from local network, but with time we do weighted sum of local network and target network weights and assign them to target network and this is known as **soft update**.
    * The soft update consists of slowly blending our regular network weights with our target weights.
        * Every step, mix  0.010.01  of regular network weights with target network weights.
### Update Critic
* In DDPG paper critic maps **Q-value**, and in critic network we input both state vector and action vector.
* So in critic network we want to output optimal **action value** and we know this from **Temporal difference algorithm** that `Q(s,a)= r + γ * max_a Q(next state,a)`.
* So in a Critic network:
    * Predicted_value : ` Q_expected = critic_local(state,action) `
    * Target_value : `r + γ * critic_target(next_state,actor_target(next_state))`
* This way of training critic network is pushing actor network to output the optimal action, in other words the action which maximize the action value.
* And one important thing to notice is that we want our critic to be somewhat bias but should have low **variance** that is the reason we use temporal difference algorithm to compute our target.
* And as Actor has a high variance, with critic's output we train Actor, the main idea is to reduce the variance in actor network so we can train the network.
* Advance algorithm like A3C,PPO all want to reduce the variance problem in RL agent.
### Update Actor
* In Actor network we input the state and get the action vector. (estimates Policy)
* In main idea behind actor training is that, we want actor network to output such action that maximizes the action value function which is estimated by critic.
* So in Actor network:
    * Loss: `-critic_local(state,actor_local(state))` -- gradient is calculated across this loss.
    * minus sign because we want to do gradient ascent.


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

3. To see a smart agent in your local machine run this code:
    


### Result and Analysis 
![Tensor board][image1]
* To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ on your web browser.
`$ tensorboard --logdir='./logs' --port=6006`
### Average Reward (Episode score)
![reward_agent][image3]
### Rolling Mean (100 episodes)
![rolling_mean][image4]


### Future Work
* Will try different algorithms like PPO, D4PG.
* Will learn how to make environment in Unity framework.
* Will try to solve more complex problem like arm with three joints.
