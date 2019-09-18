from osim.env import L2M2019Env
import numpy as np

# this is the official setting for Learn to Move 2019 #
model = '3D'
difficulty = 2
seed = None
project = True
obs_as_dict = True
# this is the official setting for Learn to Move 2019 #

env = L2M2019Env(seed=seed, difficulty=difficulty, visualize=False)
env.change_model(model=model, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=project, seed=seed, obs_as_dict=obs_as_dict)

total_reward = 0
while True:
    obs_dict, reward, done, info=env.step(env.action_space.sample(), project=project, obs_as_dict=obs_as_dict)
    total_reward += reward
    print(total_reward)
    if done:
        break
