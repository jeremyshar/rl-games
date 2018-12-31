import abc

class Game(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def result(self):
		"""Gets the result of the game e.g. for chess, -1.0 if Black wins,
		1.0 if White wins, 0.0 if the game is drawn, and None if the game
		isn't over.

		Returns:
			None if the game isn't over, or the float value of the game 
			if it is over.
		"""
		pass

	@abc.abstractmethod
	def available_actions(self):
		"""Gets the actions available to the agent in the current state.

		Returns:
			A list of the available actions. The order should be stable.
		"""
		pass

	@abc.abstractmethod
	def take_action(self, action):
		"""Mutates the current state by taking the input action. The action
		can be assumed to be available/legal.

		Args:
			action: The action to take.
		"""
		pass

	@abc.abstractmethod
	def undo_action(self):
		"""Undoes the last action taken."""
		pass

	@abc.abstractmethod
	def reset(self):
		"""Resets the game to its start state."""
		pass

	@abc.abstractmethod
	def state_shape(self):
		"""Returns the shape of the game states."""
		pass

	@abc.abstractmethod
	def state(self):
		"""Gets a representation of the current game state.

		Returns:
			The state representation.
		"""
		pass 

	@abc.abstractmethod
	def state_key(self):
		"""Gets a hashable representation of the current game state.
		Note that this key should uniquely identify the state, since states
		with the same key will be treated as the same.

		Returns:
			A hashable representation of the current state.
		"""
		pass

	@abc.abstractmethod
	def print_state(self):
		"""Prints the current state of the game."""
		pass

	@abc.abstractmethod
	def maximizer_to_act(self):
		"""Returns True if it's the maximizer's turn to act in the current state."""
		pass

	@abc.abstractmethod
	def action_space(self):
		"""Returns the total number of actions in the game."""
		pass

	@abc.abstractmethod
	def action_index(self, action):
		"""Returns the index into the action space of the given action."""
		pass

	@abc.abstractmethod
	def index_to_action(self, index):
		"""Returns the action corresponding to the given index in action space"""
		pass

	@abc.abstractmethod
	def parse_action(self, action_string):
		"""Parses an action given as user input during interactive play."""
		pass

class Agent(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def select_move(self):
		"""Returns the action to take in the current state."""
		pass

class Model(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def value(self, state):
		"""Returns the value/predicted outcome of the current state."""
		pass

	@abc.abstractmethod
	def policy(self, state):
		"""Returns a 1D array of values the size of the action space
		of probabilities."""
		pass

	@abc.abstractmethod
	def train(self, data):
		"""Updates the model with the data.

		Args:
			data: A list of (state, target_value, target_policy) tuples.
		"""
		pass