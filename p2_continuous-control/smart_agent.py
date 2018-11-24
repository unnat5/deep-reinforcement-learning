from unityagents import UnityEnvironment
import numpy as np
import sys

condition = sys.stdin.readline().split()

env = UnityEnvironment(file_name='./Reacher.app')


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

from ddpg_agent import Agent
random_seed=8
agent = Agent(33,4,random_seed)
num_agents=20
# print(action_size,state_size)

import torch
if condition[0] == "random":
    pass
else:
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    agent.actor_target.load_state_dict(torch.load('checkpoint_actor_target.pth'))
    agent.critic_target.load_state_dict(torch.load('checkpoint_critic_target.pth'))


env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions=[agent.act(states[no_agent,:]) for no_agent in range(20)]
    actions = np.array(actions).reshape(20,4)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))