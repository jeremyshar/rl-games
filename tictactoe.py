from __future__ import division

import itertools
import numpy as np

import abstract

class TicTacToe(abstract.Game):
	def __init__(self, board_size, num_in_a_row):
		# Mapping between potential game winners and outcome scores. _ is a draw.
		self.value_mapping = {'_':0, 'X':1.0, 'O':-1.0}
		self.empty = '_'
		self.winner = self.empty
		self.players = ['X', 'O']
		self.first_player_to_move = True
		self.board_size = board_size
		self.num_in_a_row = num_in_a_row
		# Initialize the board.
		self.board = [[[self.empty] for i in range(board_size)] for j in range(board_size)]
		# Columns and diagonals are just for semi-efficient indexing of the game.
		self.columns = [[self.board[j][i] for j in range(board_size)] for i in range(board_size)]
		self.diagonals = [[] for i in range((board_size*4)-2)]
		for i in range(board_size):
			for j in range(board_size):
				self.diagonals[i+j].append(self.board[i][j])
				self.diagonals[(board_size-i-1)+j+(2*board_size)-1].append(self.board[i][j])
		self.moves = []

	# Checks if the game is over, and if it is, returns the game's value. Otherwise returns None.
	def result(self):
		# Iterate through the rows, columns and diagonals and check for num_in_a_row in a row.
		for line in itertools.chain(self.board, self.columns, self.diagonals):
			last_seen = self.empty
			consecutive = 1
			for elt in line:
				if elt[0] in self.players and elt[0] == last_seen:
					consecutive += 1
					if consecutive >= self.num_in_a_row:
						self.winner = last_seen
						return self.value_mapping[self.winner]
				else:
					consecutive = 1
				last_seen = elt[0]
		# If there's no winner, check if it's a draw.
		return self.value_mapping[self.empty] if len(self.available_actions()) == 0 else None

	# Prints the current state of the game.
	def print_state(self):
		for row in self.board:
			print '|' + '|'.join([elt[0] for elt in row]) + '|'

	# Returns a list of the legal moves (flattened indices of empty squares).
	def available_actions(self):
		return [i for i in range(self.board_size**2) 
			if self.board[i//self.board_size][i%self.board_size][0] == self.empty]

	# Makes a move in the game. The move should be the flattend index of an empty space.
	def take_action(self, action):
		self.board[action//self.board_size][action%self.board_size][0] = 'X' if self.first_player_to_move else 'O'
		self.moves.append(action)
		self.first_player_to_move = not self.first_player_to_move

	# Undoes the last executed move.
	def undo_action(self):
		move = self.moves.pop()
		self.winner = self.board[move//self.board_size][move%self.board_size][0] = self.empty
		self.first_player_to_move = not self.first_player_to_move

	# Resets the game to the initial state.
	def reset(self):
		self.winner = self.empty
		self.first_player_to_move = True
		self.moves = []
		for i in range(self.board_size):
			for j in range(self.board_size):
				self.board[i][j][0] = self.empty

	def state_shape(self):
		return (self.board_size, self.board_size, 3)

	# Convenience function to transform the board state into a stack of layers.
	# The first layer is the locations of X, the second O, and the third is entirely
	# 1.0 if it's O-to-move and 0.0 if it's X-to-move.
	def state(self):
		stack = np.zeros(shape=self.state_shape())
		if not self.first_player_to_move:
			stack[:, :, 2] = 1.0
		for i in range(self.board_size):
			for j in range(self.board_size):
				if self.board[i][j][0] == 'X':
					stack[i][j][0] = 1.0
				elif self.board[i][j][0] == 'O':
					stack[i][j][1] = 1.0
		return stack

	def state_key(self):
		return self.state().tostring()

	def maximizer_to_act(self):
		return self.first_player_to_move

	# Looks up an element in the board element by a flattened index.
	# [0 1 2]
	# [3 4 5]
	# [6 7 8]
	def flat_lookup(self, index):
		return self.board[index//self.board_size][index%self.board_size][0]

	def action_space(self):
		return self.board_size*self.board_size

	def action_index(self, action):
		return action

	def index_to_action(self, index):
		return index

	def parse_action(self, action_string):
		return int(action_string)

