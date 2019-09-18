#!/usr/bin/env python

from osim.env import L2M2019Env
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse
import os

from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl

# """
# NOTE: For testing your submission scripts, you first need to ensure 
# that redis-server is running in the background
# and you can locally run the grading service by running this script : 
# https://github.com/stanfordnmbl/osim-rl//blob/master/osim/redis/service.py
# The client and the grading service communicate with each other by 
# pointing to the same redis server.
# """

# """
# Please ensure that `visualize=False`, else there might be unexpected errors 
# in your submission
# """
# env = L2M2019Env(visualize=False)
# env = FlattenDictWrapper(env, ['v_tgt_field', 'pelvis', 'r_leg', 'l_leg'])

# """
# Define evaluator end point from Environment variables
# The grader will pass these env variables when evaluating
# """
# REMOTE_HOST = os.getenv("AICROWD_EVALUATOR_HOST", "127.0.0.1")
# REMOTE_PORT = os.getenv("AICROWD_EVALUATOR_PORT", 6379)
# client = Client(
#     remote_host=REMOTE_HOST,
#     remote_port=REMOTE_PORT
# )

# # Create environment
# observation = client.env_create()

# Settings
remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "b5f5cd09cb870c14547db176596d09e5" # use your aicrowd token
# your aicrowd token (API KEY) can be found at your prorfile page at https://www.aicrowd.com

client = Client(remote_base)

# Create environment
observation = client.env_create(aicrowd_token, env_id='L2M2019Env')

"""
The grader runs N simulations of at most 1000 steps each. 
We stop after the last one
A new simulation starts when `clinet.env_step` returns `done==True`
and all the simulations end when the subsequent `client.env_reset()` 
returns a False
"""
mode = '3D'
difficulty = 2
visualize=False
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

params = np.loadtxt('./osim/control/params_3D.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
total_reward = 0
while True:
    locoCtrl.set_control_params(params)
    action = locoCtrl.update(observation)
    
    [observation, reward, done, info] = client.env_step(action)
    total_reward += reward
    print(total_reward)
    if done:
        total_reward = 0
        observation = client.env_reset()
        if not observation:
            break

client.submit()
