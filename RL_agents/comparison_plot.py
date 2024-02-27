#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dqn = pd.read_csv('dqn_100_epoch.csv')
ppo = pd.read_csv('ppo_100_epoch.csv')
a2c = pd.read_csv('a2c_100_epoch.csv')

# Assuming 'Episode' column exists that represents the episode number
# If it doesn't, you can use the index as a proxy for episode numbers
dqn_episode = np.arange(1, len(dqn) + 1)  # Assuming episodes are sequential and start at 1
ppo_episode = np.arange(1, len(ppo) + 1)
a2c_episode = np.arange(1, len(ppo) + 1)

# Extract rewards
dqn_reward = dqn['Reward']
ppo_reward = ppo['Reward']
a2c_reward = a2c['Total Reward']

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size for better readability

# Plot DQN and PPO rewards per episode
plt.plot(dqn_episode, dqn_reward, label='DQN', color='blue')
plt.plot(ppo_episode, ppo_reward, label='PPO', color='red')
plt.plot(a2c_episode, a2c_reward, label='A2C', color='green')

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN vs PPO vs A2C Performance Comparison')

# Add a legend to differentiate between DQN and PPO
plt.legend()

# Optional: Add grid for better readability
plt.grid(True)

# Show the plot
plt.show()

