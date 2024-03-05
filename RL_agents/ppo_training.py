import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from tensorflow.keras import Model, layers, optimizers
import tensorflow as tf
from collections import namedtuple, deque
import random
import pandas as pd
import scipy.signal
from ppo_agent import PPOAgent

# Assuming Actor, Critic, and PPOAgent are defined in ppo_agent.py
from ppo_model import Actor, Critic

# Set up argument parser
parser = argparse.ArgumentParser(description='NS-3 PPO Simulation Control')
parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations', type=int, default=200, help='Number of iterations (episodes), Default: 200')
args = parser.parse_args()

# Use the parsed arguments
startSim = bool(args.start)
total_episodes = args.iterations

# Environment and Training Setup
env = ns3env.Ns3Env(port=5555, stepTime=0.5, startSim=startSim, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123}, debug=False)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
state_size = ob_space.shape[0]
action_size = ac_space.n

# PPO Hyperparameters
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.9
lamda = 0.95
clip_ratio = 0.2
update_epochs = 4
minibatch_size = 64

# Initialize PPO Agent
agent = PPOAgent(state_size, action_size, actor_lr, critic_lr, gamma, lamda, clip_ratio, update_epochs, minibatch_size)

# Tracking variables
time_history = []
rew_history = []  # Keep track of rewards for plotting

max_env_steps = 120  # Maximum number of steps per episode


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Assuming 'env' and 'agent' are already defined and initialized elsewhere in the script

def ppo_training():

    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0  # Initialize step counter for the current episode

        while not done and steps < max_env_steps:
            action, value, log_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            total_reward += reward
            steps += 1  # Increment step counter

            if done or steps == max_env_steps:
                # Pass the value of the next state unless it's terminal or reached max steps
                next_value = 0 if done else agent.critic(np.array([next_state], dtype=np.float32)).numpy()[0, 0]
                agent.finish_path(next_value)

        # Update PPO agent
        agent.learn()

        print(f"Episode: {episode + 1}/{total_episodes}, Total reward: {total_reward}, Steps: {steps}")
        rew_history.append(total_reward)
        time_history.append(episode)
        
    df = pd.DataFrame(list(zip(time_history, rew_history)), columns=['Time', 'Reward'])
    df.to_csv('ppo_200.csv', index=False)
    agent.actor.save('ppo_actor_200.keras')
    agent.critic.save('ppo_critic_200.keras')
    # Close the environment
    env.close()

    # Plotting the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rew_history, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    ppo_training()

