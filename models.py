from __future__ import division

import numpy as np

import abstract

class TableModel(abstract.Model):
	"""A model for TicTacToe that explicitly stores all stats in a table.
	The states are reflection and rotation invariant.
	"""

	def __init__(self, board_size):
		"""Constructor for a TableModel.

		Args:
			board_size: An int, the size of the board.
		"""
		self.board_size = board_size
		self.table = {}

	def transform_policy(self, policy, rotation_amt, flipped, reverse_direction):
		"""Transforms a 1D policy vector, rotating its values by the rotation_amt,
		and reflecting it if specified.

		Args:
			policy: The input 1D policy array to be transformed.
			rotation_amt: An int between 0 and 3, the number of 90 degree rotations.
			flipped: A boolean, whether or not the policy should be reflected.
			reverse_direction: A boolean, whether or not the transformation should be
				done in reverse/undone.

		Returns:
			The resulting transformed policy as a 1D array.
		"""
		reshaped = np.reshape(policy, (self.board_size, self.board_size))
		if reverse_direction:
			mirrored =  np.fliplr(reshaped) if flipped else reshaped
			result = np.rot90(mirrored, k=4-rotation_amt)
		else:
			rotated = np.rot90(reshaped, k=rotation_amt)
			result = np.fliplr(rotated) if flipped else rotated
		return np.reshape(result, (-1))


	def lookup(self, state):
		"""Returns the stats for the position if it's in the table, along with the transformation
		parameters required to turn the input state into the key state. For convenience, returns
		the existing key state. If the state doesn't exist in the table, returns None.
		
		Args:
			state: The input state to look up in the table.

		Returns:
			None if the state isn't in the table, or the state's target value, target policy, 
			and weight as a tuple, along with the number of rotations required to match the
			stored state, whether a reflection was required to match the stored key, and the
			actual stored key.
		"""
		for flipped in [False, True]:
			for rotation_amt in range(4):
				rotated = np.rot90(state, k=rotation_amt, axes=(1, 2))
				symmetry = np.flip(rotated, axis=2) if flipped else rotated
				if symmetry.tostring() in self.table:
					value, policy, weight = self.table[symmetry.tostring()]
					policy = self.transform_policy(policy, rotation_amt, flipped, reverse_direction=True)
					return (value, policy, weight), rotation_amt, flipped, symmetry.tostring()
		return None


	def insert_or_update(self, state, stats):
		"""Inserts the state and stats if they don't exist, otherwise averages these stats with 
		the stats from the existing entry (using the weights).

		Args:
			state: The state to insert into the table.
			stats: The stats associated with the state.
		"""
		lookup_result = self.lookup(state)
		if lookup_result is None:
			self.table[state.tostring()] = stats
		else:
			(v, p, w), rotation_amt, flipped, key = lookup_result
			value, policy, weight = stats
			new_weight = w + weight
			new_value = (v*w + value*weight) / new_weight
			new_policy = (p*w + policy*weight) / new_weight
			new_policy = self.transform_policy(new_policy, rotation_amt, flipped, reverse_direction=False)
			self.table[key] = (new_value, new_policy, new_weight)

	def policy(self, state):
		"""Returns the policy for the inpout state.

		Args:
			state: the state for which to fetch the policy.

		Returns:
			The policy vector for the input state, or a uniform random policy if the state
			isn't in the table.
		"""
		entry = self.lookup(state)
		if entry is not None:
			return entry[0][1]
		else:
			return np.ones(self.board_size*self.board_size)/(self.board_size*self.board_size)

	def value(self, state):
		"""Returns the value of the input state.

		Args:
			state: The state for which to fetch the value.

		Returns:
			A float, the value for the input state, or 0 if the state isn't in the table.
		"""
		entry = self.lookup(state)
		if entry is not None:
			return entry[0][0]
		else:
			return 0

	def train(self, data):
		"""Updates the table from a list of input data.

		Args:
			data: A list of (state, target_value, target_policy) tuples.
		"""
		for datum in data:
			self.insert_or_update(datum[0], (datum[1], datum[2], 1.0))
