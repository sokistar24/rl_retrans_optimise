import argparse
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.models import load_model
from collections import deque
import pandas as pd
from a2c_agent import A2CAgent  # Ensure you have an A2CAgent class similar to the PPOAgent

# Assuming Actor and Critic models are defined similarly as in the PPO example
from a2c_agent import Actor, Critic
import time


def a2c_testing():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='NS-3 A2C Simulation Control')
    parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations (episodes), Default: 200')
    args = parser.parse_args()

    # Use the parsed arguments
    startSim = bool(args.start)
    total_episodes = args.iterations

    # Environment setup
    env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=startSim, simSeed=0,
                        simArgs={"--simTime": 20, "--testArg": 123}, debug=False)
    env.reset()

    ob_space = env.observation_space
    ac_space = env.action_space
    s_size = ob_space.shape[0]
    a_size = ac_space.n

    # Initialize PPO Agent (without specifying actor and critic models)
    agent = A2CAgent(state_size=s_size, action_size=a_size)

    # Load the trained models
    agent.actor = load_model('a2c_actor_200.keras')
    agent.critic = load_model('a2c_critic_200.keras')

    total_episodes = args.iterations  # Renamed for clarity
    max_env_steps = 1000  # Maximum steps per episode
    start_time = time.time()
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
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time} seconds")


if __name__ == '__main__':
    a2c_testing()
