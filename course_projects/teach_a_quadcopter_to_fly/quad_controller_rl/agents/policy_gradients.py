# policy gradient agent

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
import os
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.replay_buffer import ReplayBuffer
from quad_controller_rl.agents.actor import Actor
from quad_controller_rl.agents.critic import Critic
from quad_controller_rl.agents.ou_noise import OUNoise

class DDPG(BaseAgent):
	def __init__(self, task):
		self.task = task
		self.last_state = None
		self.last_state = None

		# Constrain state and action spaces
		self.state_size = 2  # position only
		self.action_size = 1  # force only
		self.action_low = self.task.action_space.low[0:self.action_size]
		self.action_high = self.task.action_space.high[0:self.action_size]
		print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
		    self.task.observation_space.shape, self.task.action_space.shape,
		    self.state_size, self.action_size))

		# Save episode stats
		self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
		self.stats_columns = ['episode', 'total_reward']  # specify columns to save
		self.episode_num = 1
		self.total_reward = 0.0
		print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

		# replay buffer
		self.buffer_size = 100000
		self.batch_size = 64
		self.memory = ReplayBuffer(self.buffer_size)

		# Actor (Policy) Model
		self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
		self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

		# Critic (Value) Model
		self.critic_local = Critic(self.state_size, self.action_size)
		self.critic_target = Critic(self.state_size, self.action_size)

		# Initialize target model parameters with local model parameters
		self.critic_target.model.set_weights(self.critic_local.model.get_weights())
		self.actor_target.model.set_weights(self.actor_local.model.get_weights())

		# Noise process
		self.noise = OUNoise(self.action_size)

		# Algorithm parameters
		self.gamma = 0.9  # discount factor
		self.tau = 0.001  # for soft update of target parameters

		# Load/save parameters
		self.model_dir = util.get_param('out')  # you can use a separate subdirectory for each task and/or neural net architecture
		self.model_name = "takeoff-model"
		self.model_ext = ".h5"


	def preprocess_state(self, state):
		"""Reduce state vector to relevant dimensions."""
		return np.array([state[2], state[9]])   # position and velocity


	def postprocess_action(self, action):
		"""Return complete action vector."""
		complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
		complete_action[2] = action  # linear force only
		return complete_action

	def step(self, state, reward, done):
		# Reduce state vector
		state = self.preprocess_state(state)

		# Transform state, choose action, save experience, learn, etc.
		action = self.act(state)
		
		self.total_reward += reward

		# Save experience / reward
		if self.last_state is not None and self.last_action is not None:
			self.memory.add(self.last_state, self.last_action, reward, state, done)			

		# Learn, if enough samples are available in memory
		if len(self.memory) > self.batch_size:
			experiences = self.memory.sample(self.batch_size)
			self.learn(experiences)


		if done:
			# Save model weights at regular intervals
			if self.episode_num % 10 == 0:
				self.actor_local.model.save_weights(os.path.join(self.model_dir, "{}_actor{}".format(self.model_name, self.model_ext)))
				self.critic_local.model.save_weights(os.path.join(self.model_dir, "{}_critic{}".format(self.model_name, self.model_ext)))
				print("Model weights saved at episode", self.episode_num)  # [debug]

			# Write episode stats
			print('Total reward: ', self.total_reward)
			self.write_stats([self.episode_num, self.total_reward])
			self.episode_num += 1
			self.total_reward = 0
			self.last_state = None
			self.last_action = None
		else:
			self.last_state = state
			self.last_action = action

		# Return complete action vector
		return self.postprocess_action(action)


	def act(self, states):
		"""Returns actions for given state(s) as per current policy."""
		states = np.reshape(states, [-1, self.state_size])
		actions = self.actor_local.model.predict(states)
		
		return actions + self.noise.sample()  # add some noise for exploration


	def learn(self, experiences):
		"""Update policy and value parameters using given batch of experience tuples."""
		# Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
		states = np.vstack([e.state for e in experiences if e is not None])
		actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
		rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
		dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
		next_states = np.vstack([e.next_state for e in experiences if e is not None])

		# Get predicted next-state actions and Q values from target models
		#     Q_targets_next = critic_target(next_state, actor_target(next_state))
		actions_next = self.actor_target.model.predict_on_batch(next_states)
		Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

		# Compute Q targets for current states and train critic model (local)
		Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
		self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

		# Train actor model (local)
		action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
		self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

		# Soft-update target models
		self.soft_update(self.critic_local.model, self.critic_target.model)
		self.soft_update(self.actor_local.model, self.actor_target.model)


	def soft_update(self, local_model, target_model):
		"""Soft update model parameters."""
		local_weights = np.array(local_model.get_weights())
		target_weights = np.array(target_model.get_weights())

		new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
		target_model.set_weights(new_weights)


	def write_stats(self, stats):
		"""Write single episode stats to CSV file."""
		df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
		df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))  # write header first time only		