from __future__ import division

import numpy as np

# An agent that makes completely random moves.
class RandomAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game

	def select_move(self):
		return np.random.choice(self.game.legal_moves())

# An agent that picks the first legal move.
class DumbAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game

	def select_move(self):
		return self.game.legal_moves()[0]

# An agent that plays optimally using minimax.
class MinimaxAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game
		self.cache = {}

	def select_move(self, print_debug=False):
		v, moves = self.minimax(self.game)
		if print_debug:
			print 'Value: {} Optimal moves:'.format(v), sorted(moves)
		return np.random.choice(moves)

	def minimax(self, game):
		# If this result is cached, immediately return it.
		state = game.transform_state().tostring()
		if state in self.cache:
			return self.cache[state]
		# If the game is over, return the value of the result.
		if self.game.is_game_over():
			return self.game.value_mapping[game.winner], None
		# Initialize the best value to effectively +/-infinity, and the best move to invalid.
		best_value = -10.0 if game.first_player_to_move else 10.0
		best_moves = []
		for move in self.game.legal_moves():
			# Get the value of the subtree.
			self.game.make_move(move)
			value, _ = self.minimax(self.game)
			self.game.undo_move()
			change_detector = best_value
			# Keep track of the best move.
			best_value = max(best_value, value) if game.first_player_to_move else min(best_value, value)
			if change_detector != best_value:
				best_moves = [move]
			elif value == best_value:
				best_moves.append(move)
		if len(game.moves) < 4:
			self.cache[state] = (best_value, best_moves)
		return best_value, best_moves

	def optimal_moves(self):
		_, moves = self.minimax(self.game)
		return moves