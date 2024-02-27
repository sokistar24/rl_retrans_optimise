import numpy as np
import tensorflow as tf
from ppo_model import Actor, Critic  # Assuming similar structure as provided models
import scipy.signal

class A2CAgent:
    def __init__(self, state_size, action_size, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, update_steps=5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.update_steps = update_steps  # Number of steps before updating the model

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Temporary storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store_transition(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.actor(state)
        action_probs = tf.nn.softmax(logits)

        # Check for NaN in action probabilities and handle it
        if np.isnan(action_probs.numpy()).any():
            print("NaN detected in action probabilities:", action_probs.numpy())
            # Replace NaN with a uniform distribution as a fallback
            safe_action_probs = np.nan_to_num(action_probs.numpy(), nan=1.0 / self.action_size)
            action = np.random.choice(self.action_size, p=safe_action_probs[0])
        else:
            action = np.random.choice(self.action_size, p=action_probs.numpy()[0])

        return action

    def learn(self):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute returns
        returns = self.compute_returns(rewards, dones)

        # Convert lists to numpy arrays for TensorFlow compatibility
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            logits = self.actor(states)
            action_probs = tf.nn.softmax(logits)
            indices = tf.range(0, tf.shape(logits)[0]) * tf.shape(logits)[1] + actions
            chosen_log_probs = tf.gather(tf.reshape(tf.math.log(action_probs), [-1]), indices)

            values = tf.squeeze(self.critic(states), 1)
            advantages = returns - values
            actor_loss = -tf.reduce_mean(chosen_log_probs * advantages)
            critic_loss = tf.reduce_mean((returns - values) ** 2)

        actor_grads = tape_a.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape_c.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.reset_buffers()

    def compute_returns(self, rewards, dones):
        returns = np.zeros_like(rewards)
        next_return = 0
        for t in reversed(range(len(rewards))):
            next_return = rewards[t] + self.gamma * next_return * (1 - dones[t])
            returns[t] = next_return
        return returns

