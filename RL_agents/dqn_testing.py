#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from ns3gym import ns3env
from dqn_Agent import Agent,QNetwork

# Set up argument parser
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=200, help='Number of iterations, Default: 200')
args = parser.parse_args()
startSim = bool(args.start)
total_episodes = args.iterations  # Renamed for clarity
# Initialize NS-3 environment

env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=startSim, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123}, debug=False)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
s_size = ob_space.shape[0]  # Get state size from observation space
a_size = ac_space.n  # Get action size from action space

# Load the trained DQN model

# Tracking variables for rewards
rew_history = []



# Initialize Agent
agent = Agent(state_size=s_size, action_size=a_size, seed=0)

agent.qnetwork_local=load_model('dqn_model.keras') 

max_env_steps = 1000  # Set the maximum number of steps per episode
for e in range(total_episodes):
    state = env.reset()
    state = np.reshape(state, [1, s_size])
    total_reward = 0

    for time in range(max_env_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, s_size])

        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    # Possibly update the target network here, depending on your Agent class implementation
    print(f"Episode: {e + 1}/{total_episodes}, Total reward: {total_reward}")

   

# Close the environment at the end of testing
env.close()

