import argparse
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
import tensorflow as tf
from collections import deque
import pandas as pd
from a2c_agent import A2CAgent  # Ensure you have an A2CAgent class similar to the PPOAgent

# Assuming Actor and Critic models are defined similarly as in the PPO example
from a2c_agent import Actor, Critic

# Set up argument parser
parser = argparse.ArgumentParser(description='NS-3 A2C Simulation Control')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=200, help='Number of iterations (episodes), Default: 200')
args = parser.parse_args()

# Use the parsed arguments
startSim = bool(args.start)
total_episodes = args.iterations

# Environment setup
env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=startSim, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123},
                    debug=False)
env.reset()

# Define state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# A2C Hyperparameters
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.9
update_steps = 4

# Initialize PPO Agent
agent = A2CAgent(state_size, action_size, actor_lr, critic_lr, gamma, update_steps=update_steps)

# Tracking variables
time_history = []
rew_history = []  # Keep track of rewards for plotting

max_env_steps = 120  # Maximum number of steps per episode


def a2c_training():
    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0  # Initialize step counter for the current episode

        while not done and steps < max_env_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, done)

            state = next_state
            total_reward += reward
            steps += 1  # Increment step counter

            if steps % agent.update_steps == 0 or done:
                agent.learn()
                agent.reset_buffers()

            if done:
                break

        print(f"Episode: {episode + 1}/{total_episodes}, Total reward: {total_reward}, Steps: {steps}")
        rew_history.append(total_reward)
        time_history.append(episode)

    agent.actor.save('a2c_actor_200.keras')
    agent.critic.save('a2c_critic_200.keras')

    df = pd.DataFrame(list(zip(time_history, rew_history)), columns=['Episode', 'Reward'])
    df.to_csv('a2c_200.csv', index=False)
    # Close the environment
    env.close()

    # Plotting the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rew_history, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C Training Performance')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    a2c_training()
