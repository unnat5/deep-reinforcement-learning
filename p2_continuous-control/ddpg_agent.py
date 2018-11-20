import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## code is inspired from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

##========== HYPERPARAMETER ============##
BUFFER_SIZE = int(1e5)    # replay buffer
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99              # discounting factor
TAU = 1e-3                # soft update of traget parameters
LR_ACTOR = 1e-3           # learning rate for actor
LR_CRITIC = 1e-3          # learning rate for critic
WEIGHT_DECAY = 0.       # L2 weight weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment"""
    
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size 
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
        
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size,action_size,random_seed).to(device)
        self.critic_target = Critic(state_size,action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)
        
        
        # Noise process
        self.noise = OUNoise(action_size,random_seed)
        
        # Replay Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.counter = 0
        
       # Make sure target is with the same weight as the source found on slack
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward 
        for state,action,reward,next_state,done in zip(state, action, reward, next_state, done):
            self.memory.add(state, action, reward, next_state, done)
            self.counter+=1
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.counter%10==0: 
            experience = self.memory.sample()
            self.learn(experience, GAMMA)
            
    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy."""
        #Save experience / reward
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experience, gamma):
        """Update policy and value parameters using given batch of experience tuples
        
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        
        where:
            actor_target(state) -> action
            critic_target(state,action) -> Q-value
            
        Params
        ======
            experience (Torch[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        state, action, reward, next_state, done = experience
        
        # ============================== Update Critic =================================#
        # Get predicted next-state actions and Q values from target models
        
        self.actor_target.eval() ## there is no point is saving gradient
        self.critic_target.eval()
        
        actions_next = self.actor_target(next_state)
        Q_target_next = self.critic_target(next_state,actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (gamma*Q_target_next*(1-done))
        
        ## Compute Critic Loss
        Q_expected = self.critic_local(state,action)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        ## Minize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        
        # ============================== Update Actor =================================#
        ## Compute actor loss
        action_pred  = self.actor_local(state)
        actor_loss = -(self.critic_local(state,action_pred).mean())
        ## Calculating Advantage!!
        #print(Q_targets.size(),self.critic_local(state,action_pred).size())
#         actor_loss = -(torch.mean(Q_targets-self.critic_local(state,action_pred)))
        ## The reason we can calculate loss this way and we don't have
        ## to collect trajector ( noisy Monte carlo estimation; cum_reward/reward_future)
        ## is action space is continuous and differentiable and we calculate
        ## gradient w.r.t to q_value which is estimated by CRITIC.
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
#         del actor_loss
        self.actor_optimizer.step()
        
        
        # ========================== Update target network =================================#

        self.soft_update(self.critic_local,self.critic_target,TAU)
        self.soft_update(self.actor_local,self.actor_target,TAU)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param,local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            ## add noise to weights
#             local_param.data.copy_(local_param.data + self.noise.sample()[3])
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, seed, mu=0.01, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu*np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta*(self.mu-x) + self.sigma*np.array([random.gauss(0., 1.) for i in range(len(x))])
        self.state =x +dx
        return self.state
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
