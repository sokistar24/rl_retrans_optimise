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
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations, Default: 20')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

# Initialize NS-3 environment
port = 5555
stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": 20, "--testArg": 123}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()


def get_state_index(state):
    """
    Convert state to an index for the Q-table.
    Assuming state is a NumPy array like [1, 0, 0, 0], find the index of the first '1'.
    """
    return np.argmax(state)  # Returns the index of the first occurrence of the maximum value


# Q-learning parameters
learning_rate = 0.01
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.05
max_env_steps = 100

# Initialize Q-table
num_states = env.observation_space.n  # Assuming four possible states representing channel occupancy
num_actions = env.action_space.n
q_table = np.zeros([num_states, num_actions])

# Tracking variables
print(q_table)
time_history = []
rew_history = []


def epsilon_greedy_action_selection(epsilon, Q, state):
    random_number = np.random.random()
    if random_number > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = env.action_space.sample()
    return action


for e in range(iterationNum):
    state = env.reset()
    state_value = int(state[0])  # Convert initial state to an integer value

    rewardsum = 0
    for time in range(max_env_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_value])

        next_state, reward, done, info = env.step(action)
        next_state_value = int(next_state[0])  # Convert next state to an integer value
        print(f"Next state: {next_state}, Index: {next_state_value}")

        # Q-learning update
        old_value = q_table[state_value, action]
        next_max = np.max(q_table[next_state_value])
        new_value = old_value + learning_rate * (reward + gamma * next_max - old_value)
        q_table[state_value, action] = new_value

        state_value = next_state_value  # Update state for the next iteration
        rewardsum += reward

        if done:
            break

    # Reduce epsilon (exploration rate)
    epsilon = max(epsilon_min, epsilon - epsilon_decay)

    time_history.append(time)
    rew_history.append(rewardsum)
    print(f"Episode: {e + 1}/{iterationNum}, Reward: {rewardsum}")

env.close()

# Output the Q-table for inspection
print("Q-Table:")
print(q_table)

# Plotting the learning performance
plt.title('q_Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")
plt.plot(range(len(rew_history)), rew_history, label='Reward', linestyle="-")
plt.xlabel('Episode')
plt.ylabel('Time/Reward')
plt.legend()
plt.show()



