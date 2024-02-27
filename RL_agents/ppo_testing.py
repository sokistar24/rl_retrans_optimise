import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import namedtuple, deque
import random
import pandas as pd
import scipy.signal
from ppo_agent import PPOAgent

# Assuming Actor, Critic, and PPOAgent are defined in ppo_agent.py
from ppo_agent import PPOAgent, Actor, Critic


# Set up argument parser
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=200, help='Number of iterations, Default: 200')
args = parser.parse_args()
startSim = bool(args.start)
# Set up NS-3 environment (similar to previous examples)
env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=True, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123}, debug=False)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
s_size = ob_space.shape[0]
a_size = ac_space.n

# Initialize PPO Agent (without specifying actor and critic models)
agent = PPOAgent(state_size=s_size, action_size=a_size)

# Load the trained models
agent.actor = load_model('ppo_actor_model.keras')
agent.critic = load_model('ppo_critic_model.keras')

total_episodes = args.iterations  # Renamed for clarity
max_env_steps = 1000  # Maximum steps per episode

for e in range(total_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_env_steps:
        state = np.reshape(state, [1, s_size])
        # Use the loaded actor model to select an action
        action_probs = agent.actor.predict(state)
        action = np.argmax(action_probs)  # Assuming deterministic action selection for simplicity

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        if done:
            break

    print(f"Test Episode: {e + 1}/{total_episodes}, Total reward: {total_reward}")

# Close the environment
env.close()

