from __future__ import division

import numpy as np

class RandomAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game

	def select_move(self):
		return np.random.choice(self.game.legal_moves())

class DumbAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game

	def select_move(self):
		return self.game.legal_moves()[0]

class MinimaxAgent(object):
	def __init__(self, name, game):
		self.name = name
		self.game = game

	def select_move(self):
		_, move = self.minimax(self.game)
		return move

	def minimax(self, game):
		if self.game.is_game_over():
			return self.game.value_mapping[game.winner], None
		best_value = -10.0 if game.first_player_to_move else 10.0
		best_move = -1
		for move in self.game.legal_moves():
			self.game.make_move(move)
			value, _ = self.minimax(self.game)
			self.game.undo_move()
			change_detector = best_value
			best_value = max(best_value, value) if game.first_player_to_move else min(best_value, value)
			if change_detector != best_value:
				best_move = move
		return best_value, best_move