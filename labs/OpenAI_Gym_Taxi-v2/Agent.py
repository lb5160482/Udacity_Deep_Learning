import numpy as np
from collections import defaultdict

class Agent:

	def __init__(self, nA=6):
		""" Initialize agent.

		Params
		======
		- nA: number of actions available to the agent
		"""
		self.nA = nA
		self.Q = defaultdict(lambda: np.zeros(self.nA))
		self.epsilon = 1
		self.episode = 1
		self.action = 0
		self.next_action = 0
		self.alpha = 0.01
		self.gamma = 1.0

	def select_action(self, state):
		""" Given the state, select an action.

		Params
		======
		- state: the current state of the environment

		Returns
		=======
		- action: an integer, compatible with the task's action space
		"""
		action_probs = self.get_probs(state)
		self.action = np.random.choice(np.arange(self.nA), p=action_probs)
		
		return self.action

	# Q-Learning
	def step(self, state, action, reward, next_state, done):
		""" Update the agent's knowledge, using the most recently sampled tuple.

		Params
		======
		- state: the previous state of the environment
		- action: the agent's previous choice of action
		- reward: last reward received
		- next_state: the current state of the environment
		- done: whether the episode is complete (True or False)
		"""
		next_action_probs = self.get_probs(next_state)
		self.next_action = np.random.choice(np.arange(self.nA), p=next_action_probs)
		self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
		if done:
			self.action = 0
			self.next_action = 0
			self.episode += 1
			self.epsilon = 1.0 / self.episode

	def get_probs(self, state):
	    action_probs = np.ones(self.nA) * self.epsilon / self.nA
	    best_action = np.argmax(self.Q[state])
	    action_probs[best_action] = 1 - self.epsilon + self.epsilon / self.nA
	    
	    return action_probs