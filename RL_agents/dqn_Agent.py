
from tensorflow.keras import Model, layers, optimizers
import tensorflow as tf
from collections import namedtuple, deque
import random
import pandas as pd
import numpy as np

GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4
BATCH_SIZE = 64

# Custom QNetwork Class
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=24):
        super(QNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.fc1 = layers.Dense(fc1_units, activation='relu')
        self.fc2 = layers.Dense(fc2_units, activation='relu')
        self.fc3 = layers.Dense(action_size)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

# Agent Class
class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optimizers.Adam(learning_rate=5e-4)

        # Replay memory
        self.memory = ReplayBuffer(action_size, int(1e5), 64, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action_values = self.qnetwork_local(state)
        if random.random() > eps:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        future_rewards = self.qnetwork_target(next_states)
        max_future_rewards = tf.reduce_max(future_rewards, axis=1, keepdims=True)
        updated_q_values = rewards + gamma * max_future_rewards * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.qnetwork_local(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, tf.one_hot(actions, depth=self.action_size)), axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(updated_q_values - q_action))

        gradients = tape.gradient(loss, self.qnetwork_local.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.qnetwork_local.trainable_variables))

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(local_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau) * target_weights[i]
        target_model.set_weights(target_weights)

# ReplayBuffer Class
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences]).astype(np.int32)
        rewards = np.vstack([e.reward for e in experiences]).astype(np.float32)
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)






