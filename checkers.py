from __future__ import division

import numpy as np

import abstract

class Checkers(abstract.Game):
	def __init__(self, board_size, rows_per_player):
		"""Constructor for Checkers.

		Args:
			board_size: An even integer, the side length of the board.
			rows_per_player: The number of rows of pieces each player has.
				Expected to be less than half board_size.
		"""
		# Mapping between potential game winners and outcome scores. _ is a draw.
		self.value_mapping = {'_':0, 'X':1.0, 'O':-1.0}
		self.empty = '_'
		self.impassable = '_'
		self.winner = self.empty
		self.players = ['X', 'O']
		self.first_player_to_move = True
		self.board_size = board_size
		if board_size % 2 == 1:
			raise ValueError('Please use an even number for board_size.')
		if rows_per_player >= board_size // 2:
			raise ValueError('Please use a board_size more than double rows_per_player')
		self.rows_per_player = rows_per_player
		self.active_capturer = None
		# Initialize the board.
		self.board = [[[self.empty] for i in range(board_size)] for j in range(board_size)]
		self.reset()
		self.action_indices = {}
		self.indices_to_actions = {}
		self.possible_actions = 0
		for r in range(board_size):
			for c in range(board_size):
				if (r + c) % 2 == 0:
					continue
				for rdelta in [v for v in [-1, 1] if r+v >= 0 and r+v < board_size]:
					for cdelta in [v for v in [-1, 1] if c+v >= 0 and c+v < board_size]:
						self.action_indices[(r, c, r+rdelta, c+cdelta)] = self.possible_actions
						self.indices_to_actions[self.possible_actions] = (r, c, r+rdelta, c+cdelta)
						self.possible_actions += 1
				for rdelta in [v for v in [-2, 2] if r+v >= 0 and r+v < board_size]:
					for cdelta in [v for v in [-2, 2] if c+v >= 0 and c+v < board_size]:
						self.action_indices[(r, c, r+rdelta, c+cdelta)] = self.possible_actions
						self.indices_to_actions[self.possible_actions] = (r, c, r+rdelta, c+cdelta)
						self.possible_actions += 1

	def state_shape(self):
		return (self.board_size, self.board_size//2, 5)

	# Convenience function to transform the board state into a stack of layers.
	def state(self):
		s = np.zeros(shape=self.state_shape())
		for r in xrange(self.board_size):
			for c in xrange(self.board_size):
				if self.board[r][c][0] == self.empty or self.board[r][c][0] == self.impassable:
					continue
				elif self.board[r][c][0] == 'x':
					s[r][c//2][0] = 1.0
				elif self.board[r][c][0] == 'X':
					s[r][c//2][1] = 1.0
				elif self.board[r][c][0] == 'o':
					s[r][c//2][2] = 1.0
				elif self.board[r][c][0] == 'O':
					s[r][c//2][3] = 1.0
		if self.maximizer_to_act:
			s[:, :, 4] = 1.0
		return s

	def state_key(self):
		return self.state().tostring()

	# Checks if the game is over, and if it is, returns the game's value. Otherwise returns None.
	def result(self):
		if self.state_key() in self.state_keys:
			return self.value_mapping[self.winner]
		still_alive = set()
		for row in self.board:
			for col in row:
				if col[0].upper() in self.players:
					still_alive.add(col[0].upper())
		if len(still_alive) == 1:
			self.winner = still_alive.pop()
			return self.value_mapping[self.winner]
		if not self.available_actions():
			if self.first_player_to_move:
				self.winner = self.players[1]
			else:
				self.winner = self.players[0]
			return self.value_mapping[self.winner]
		return None

	# Prints the current state of the game.
	def print_state(self):
		print '   ' + ' '.join(map(str, range(self.board_size)))
		for row in range(self.board_size):
			print str(row) + ' |' + '|'.join([elt[0] for elt in self.board[row]]) + '|'

	def recalculate_available_actions(self):
		self.actions = [[], [], [], []]
		for row in range(self.board_size):
			for col in range(self.board_size):
				if self.board[row][col][0] in ('x', 'O', 'X') and row < (self.board_size - 1):
					for dest_col in [c for c in [col - 1, col + 1] if c < self.board_size and c >= 0 and self.board[row+1][c][0] == self.empty]:
						action_type = 2 if self.board[row][col][0] == 'O' else 0
						self.actions[action_type].append((row, col, row+1, dest_col))
				if self.board[row][col][0] in ('o', 'X', 'O') and row > 0:
					for dest_col in [c for c in [col - 1, col + 1] if c < self.board_size and c >= 0 and self.board[row-1][c][0] == self.empty]:
						action_type = 0 if self.board[row][col][0] == 'X' else 2
						self.actions[action_type].append((row, col, row-1, dest_col))
				if self.board[row][col][0] in ('x', 'X') and row < (self.board_size - 2):
					for dest_col in [c for c in [col - 2, col + 2] if c < self.board_size and c >= 0 and self.board[row+1][(c + col)//2][0] in ('o', 'O') and self.board[row+2][c][0] == self.empty]:
						self.actions[1].append((row, col, row+2, dest_col))
				if self.board[row][col][0] in ('o', 'O') and row > 1:
					for dest_col in [c for c in [col - 2, col + 2] if c < self.board_size and c >= 0 and self.board[row-1][(c + col)//2][0] in ('x', 'X') and self.board[row-2][c][0] == self.empty]:
						self.actions[3].append((row, col, row-2, dest_col))
				if self.board[row][col][0] == 'X' and row > 1:
					for dest_col in [c for c in [col - 2, col + 2] if c < self.board_size and c >= 0 and self.board[row-1][(c + col)//2][0] in ('o', 'O') and self.board[row-2][c][0] == self.empty]:
						self.actions[1].append((row, col, row-2, dest_col))
				if self.board[row][col][0] == 'O' and row < (self.board_size - 2):
					for dest_col in [c for c in [col - 2, col + 2] if c < self.board_size and c >= 0 and self.board[row+1][(c + col)//2][0] in ('x', 'X') and self.board[row+2][c][0] == self.empty]:
						self.actions[3].append((row, col, row+2, dest_col))

	# Returns a list of the legal moves.
	def available_actions(self):
		self.recalculate_available_actions()
		if self.active_capturer is not None:
			action_type = 1 if self.first_player_to_move else 3
			return [a for a in self.actions[action_type] if a[0] == self.active_capturer[0] and a[1] == self.active_capturer[1]] 
		if self.first_player_to_move:
			if self.actions[1]:
				return self.actions[1]
			else:
				return self.actions[0]
		else:
			if self.actions[3]:
				return self.actions[3]
			else:
				return self.actions[2]

	# Makes a move in the game.
	def take_action(self, action):
		self.state_keys.add(self.state_key())
		placeholder = self.board[action[0]][action[1]][0]
		self.board[action[0]][action[1]][0] = self.empty
		made_king = False
		if self.first_player_to_move and placeholder.islower() and action[2] == self.board_size-1:
			made_king = True
			self.kings.append(len(self.moves))
			placeholder = placeholder.upper()
		if (not self.first_player_to_move) and placeholder.islower() and action[2] == 0:
			made_king = True
			self.kings.append(len(self.moves))
			placeholder = placeholder.upper()
		self.board[action[2]][action[3]][0] = placeholder
		continue_capturing = False
		if abs(action[0] - action[2]) == 2:
			continue_capturing = True
			self.captures[len(self.moves)] = self.board[(action[0] + action[2])//2][(action[1] + action[3])//2][0]
			self.board[(action[0] + action[2])//2][(action[1] + action[3])//2][0] = self.empty
			self.active_capturer = [action[2], action[3]]
		if continue_capturing and not made_king:
			self.recalculate_available_actions()
			action_type = 1 if self.first_player_to_move else 3
			further_captures = [a for a in self.actions[action_type] if a[0] == self.active_capturer[0] and a[1] == self.active_capturer[1]]
			if len(further_captures) == 0:
				continue_capturing = False
		if made_king or not continue_capturing:
			self.active_capturer = None
			self.first_player_to_move = not self.first_player_to_move
			self.turn_changes.append(len(self.moves))
		self.moves.append(action)

	# Undoes the last executed move.
	def undo_action(self):
		draw = self.result() == self.value_mapping[self.empty]
		action = self.moves.pop()
		if len(self.moves) in self.captures:
			self.board[(action[0] + action[2])//2][(action[1] + action[3])//2][0] = self.captures[len(self.moves)]
			del self.captures[len(self.moves)]
			self.active_capturer = [action[0], action[1]]
		placeholder = self.board[action[2]][action[3]][0]
		self.winner = self.board[action[2]][action[3]][0] = self.empty
		if len(self.moves) in self.kings:
			placeholder = placeholder.lower()
			self.kings.pop()
		self.board[action[0]][action[1]][0] = placeholder
		if len(self.moves) in self.turn_changes:
			self.turn_changes.pop()
			self.active_capturer = None
			self.first_player_to_move = not self.first_player_to_move
		if not draw:
			self.state_keys.remove(self.state_key())

	# Resets the game to the initial state.
	def reset(self):
		self.winner = self.empty
		self.first_player_to_move = True
		self.moves = []
		self.turn_changes = []
		self.captures = {}
		self.kings = []
		self.state_keys = set()
		self.active_capturer = None
		for row in range(self.board_size):
			for col in range(self.board_size):
				if (row + col) % 2 == 0:
					self.board[row][col][0] = self.impassable
				elif row < self.rows_per_player:
					self.board[row][col][0] = 'x'
				elif self.board_size - row <= self.rows_per_player:
					self.board[row][col][0] = 'o'
				else:
					self.board[row][col][0] = self.empty
		self.recalculate_available_actions()

	def maximizer_to_act(self):
		return self.first_player_to_move

	def action_space(self):
		return self.possible_actions

	def action_index(self, action):
		return self.action_indices[action]

	def index_to_action(self, index):
		return self.indices_to_actions[index]

	def parse_action(self, action_string):
		return tuple(int(x) for x in action_string.split(','))

