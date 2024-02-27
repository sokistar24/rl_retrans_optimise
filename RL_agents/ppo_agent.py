import numpy as np
import tensorflow as tf
from ppo_model import Actor, Critic 
import scipy.signal

class PPOAgent:
    def __init__(self, state_size, action_size, actor_lr=1e-2, critic_lr=2e-2, gamma=0.99, lamda=0.95, clip_ratio=0.2, update_epochs=4, minibatch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lamda = lamda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Experience buffer
        self.states = []
        self.actions = []
        self.advantages = []
        self.log_probs = []
        self.rewards = []
        self.returns = []
        self.values = []
        self.dones = []

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.advantages = []
        self.log_probs = []
        self.rewards = []
        self.returns = []
        self.values = []
        self.dones = []

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def finish_path(self, last_value=0):
        rewards = np.append(self.rewards, last_value)
        values = np.append(self.values, last_value)
        # Compute returns and advantages
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.returns = self.discount_cumsum(rewards, self.gamma)[:-1]
        self.advantages = self.discount_cumsum(deltas, self.gamma * self.lamda)

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.actor(state)
        action_probs = tf.nn.softmax(logits)
        action = np.random.choice(self.action_size, p=action_probs.numpy()[0])

        value = self.critic(state)
        log_prob = tf.math.log(action_probs[0, action])

        return action, value.numpy()[0, 0], log_prob.numpy()

    def learn(self):
        # Convert lists to numpy arrays for TensorFlow compatibility
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        returns = np.array(self.returns, dtype=np.float32)
        advantages = np.array(self.advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Normalize advantages

        # Optimization
        for _ in range(self.update_epochs):
            # Update actor and critic
            with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                logits = self.actor(states)
                action_probs = tf.nn.softmax(logits)
                indices = tf.range(0, tf.shape(logits)[0]) * tf.shape(logits)[1] + actions
                chosen_log_probs = tf.gather(tf.reshape(tf.math.log(action_probs), [-1]), indices)

                ratios = tf.exp(chosen_log_probs - np.array(self.log_probs))
                clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

                values = tf.squeeze(self.critic(states), 1)
                critic_loss = tf.reduce_mean((returns - values) ** 2)

            actor_grads = tape_a.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape_c.gradient(critic_loss, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.reset_buffers()

