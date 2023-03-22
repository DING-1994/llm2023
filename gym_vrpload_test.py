#from gym import envs
#print(envs.registry.all())
#from gym import register
import gym
import time
import networkx as nx
import numpy as np
from hrs_hot_file import hrs_hot_func

name="vrp-{0}agent_num-v2".format( 2, )
print("name",name)
env=gym.make(name,state_repre_flag='coordinate')
n_obs=env.reset()
print("n_obs",n_obs,type(n_obs),)
n_actions = env.action_space[0].n
obs_size = env.observation_space[0].low.size


print("action_space",env.action_space)
print("observation_space",env.observation_space)


for _ in range(50):
    #env.render()
    #time.sleep(1)

    hrs_hot=hrs_hot_func(env,n_obs)
    #print("n_obs",n_obs,)
    print("one_hot",env.obs_onehot)
    #print("hrs_hot",hrs_hot)
    print(env.current_start,env.current_goal)
    

    actions=env.action_space.sample()
    n_obs, reward, done, info = env.step(actions)

    print("actions",actions,"reward",reward, done)

    

    #print("current_start",env.current_start)



