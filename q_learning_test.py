#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env

# Set up argument parser
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=1, help='Number of iterations, Default: 20')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

# Initialize NS-3 environment
port = 5555
#stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": 20, "--testArg": 123}
debug = False

# Initialize NS-3 environment for testing
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()

# Load the trained Q-table
q_table = np.load('q_table_dynamic_reward_200_epoch.npy')

# Tracking variables
time_history = []
rew_history = []
max_env_steps = 200080000
# Test the model
for e in range(iterationNum):
    state = env.reset()
    state_value = int(state[0])  # Convert initial state to an integer value

    rewardsum = 0
    for time in range(max_env_steps):
        # Select action based on Q-table
        action = np.argmax(q_table[state_value])

        # Execute action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        next_state_value = int(next_state[0])

        # Update state and accumulate reward
        state_value = next_state_value
        rewardsum += reward

        if done:
            break

    # Store test results
    time_history.append(time)
    rew_history.append(rewardsum)
    print(f"Test Episode: {e + 1}/{iterationNum}, Reward: {rewardsum}")

# Close the environment
env.close()

# Plotting the learning performance




