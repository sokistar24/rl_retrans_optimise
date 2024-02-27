import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.logits = layers.Dense(action_size, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.logits(x)

class Critic(tf.keras.Model):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size

        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.values = layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.values(x)


