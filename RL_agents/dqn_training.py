import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from tensorflow.keras import Model, layers, optimizers
import tensorflow as tf
from collections import namedtuple, deque
from dqn_Agent import Agent,QNetwork
import random
import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser(description='NS-3 DQN Simulation Control')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=200, help='Number of iterations (episodes), Default: 200')
args = parser.parse_args()

# Use the parsed arguments
startSim = bool(args.start)
total_episodes = args.iterations  # Renamed for clarity

max_env_steps = 120

# Environment and Training Setup
# Simplified for brevity - initialize your NS3 environment and training loop
env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=startSim, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123}, debug=False)
env.reset()
ob_space = env.observation_space
ac_space = env.action_space
s_size = ob_space.shape[0] #env.observation_space.n #
a_size = ac_space.n
# Hyperparameters and Constants
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4
BATCH_SIZE = 64
lr=1e-4


# Initialize Agent
agent = Agent(state_size=s_size, action_size=a_size,lr=lr, seed=0)

# Initialize NS-3 environment with command-line arguments

# Assuming the Agent class has been defined and instantiated as `agent`
# and total_episodes, max_env_steps, epsilon_start, epsilon_end, epsilon_decay are defined

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
# Tracking variables
time_history = []
rew_history = []

def dqn_training():

    for episode in range(total_episodes):
        state = env.reset()
        state = np.reshape(state, [1, s_size])
        total_reward = 0
        done = False
        steps = 0  # Initialize step counter for the current episode

        while not done and steps < max_env_steps:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, s_size])

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1  # Increment step counter

            if done:
                break

            # Epsilon decay
            epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Possibly update the target network here, depending on your Agent class implementation
        print(f"Episode: {episode + 1}/{total_episodes}, Total reward: {total_reward}, Steps: {steps}")

        # Track rewards for plotting
        rew_history.append(total_reward)
        time_history.append(episode)

    df = pd.DataFrame(list(zip(time_history, rew_history)), columns=['Time', 'Reward'])
    df.to_csv('dqn_150.csv', index=False)
    agent.qnetwork_local.save('dqn_150.keras')  # TensorFlow will infer the SavedModel format
    # Close the environment
    env.close()

    # Plotting the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rew_history, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training lr=0.01')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    dqn_training()
