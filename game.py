from __future__ import division

import itertools
import numpy as np

class TicTacToe(object):
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

	# Checks if the game is over, and if it is, sets the game's winner.
	def is_game_over(self):
		# Iterate through the rows, columns and diagonals and check for 3 in a row.
		for line in itertools.chain(self.board, self.columns, self.diagonals):
			last_seen = self.empty
			consecutive = 1
			for elt in line:
				if elt[0] in self.players and elt[0] == last_seen:
					consecutive += 1
					if consecutive >= self.num_in_a_row:
						self.winner = last_seen
						return True
				else:
					consecutive = 1
				last_seen = elt[0]
		# If there's no winner, check if it's a draw.
		return len(self.legal_moves()) == 0

	# Prints the current state of the game.
	def print_board(self):
		for row in self.board:
			print '|' + '|'.join([elt[0] for elt in row]) + '|'

	# Returns a list of the legal moves (flattened indices of empty squares).
	def legal_moves(self):
		return [i for i in range(self.board_size**2) 
			if self.board[i//self.board_size][i%self.board_size][0] == self.empty]

	# Makes a move in the game. The move should be the flattend index of an empty space.
	def make_move(self, move):
		self.board[move//self.board_size][move%self.board_size][0] = 'X' if self.first_player_to_move else 'O'
		self.moves.append(move)
		self.first_player_to_move = not self.first_player_to_move

	# Undoes the last executed move.
	def undo_move(self):
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

	# Convenience function to transform the board state into a stack of layers.
	# The first layer is the locations of X, the second O, and the third is entirely
	# 1.0 if it's O-to-move and 0.0 if it's X-to-move.
	def transform_state(self):
		stack = np.zeros([self.board_size, self.board_size, 3])
		if not self.first_player_to_move:
			stack[:, :, 2] = np.ones([self.board_size, self.board_size])
		for i in range(self.board_size):
			for j in range(self.board_size):
				if self.board[i][j][0] == 'X':
					stack[i][j][0] = 1.0
				elif self.board[i][j][0] == 'O':
					stack[i][j][1] = 1.0
		return stack


	# Looks up an element in the board element by a flattened index.
	# [0 1 2]
	# [3 4 5]
	# [6 7 8]
	def flat_lookup(self, index):
		return self.board[index//self.board_size][index%self.board_size][0]

	# Convenience function to represent the current state as features. We construct
	# features for each potential move (legal or not), and the model should share
	# its weights. The features are player agnostic, based on who's to move.
	# 0: Whether the space is the center.
	# 1: Whether the space is a corner.
	# 2: Whether the space is an edge.
	# 3: Whether the space completes a 3-in-a-row for the current player.
	# 4: Whether the space blocks a 2 of the opponent.
	# 5: Whether the space creates a 2 for the current player.
	# 6: Whether the space creates multiple 2s for the current player.
	# 7: Whether the space blocks multiple 1s for the opponent.
	# 8: Whether the space is occupied.
	def make_features(self):
		space_mapping = {
			0: [(1, 2), (3, 6), (4, 8)],
			1: [(0, 2), (4, 7)],
			2: [(0, 1), (5, 8), (4, 6)],
			3: [(4, 5), (0, 6)],
			4: [(3, 5), (1, 7), (0, 8), (2, 6)],
			5: [(3, 4), (2, 8)],
			6: [(7, 8), (0, 3), (2, 4)],
			7: [(6, 8), (1, 4)],
			8: [(6, 7), (2, 5), (0, 4)]
		}
		current = 'X' if self.first_player_to_move else 'O'
		features = np.zeros([9, 9])
		for space in range(9):
			if space == 4:
				features[space][0] = 1.0
			elif space in [0, 2, 6, 8]:
				features[space][1] = 1.0
			else:
				features[space][2] = 1.0
			if self.flat_lookup(space) == '_':
				features[space][8] = 1.0
			twos_created = 0
			ones_blocked = 0
			for pair in space_mapping[space]:
				a = self.flat_lookup(pair[0])
				b = self.flat_lookup(pair[1])
				if a == b and a == current:
					features[space][3] = 1.0
				if a == b and a != self.empty and a != current:
					features[space][4] = 1.0
				if a == current and b == self.empty:
					twos_created += 1
				if b == current and a == self.empty:
					twos_created += 1
				if a != current and a != self.empty and b == self.empty:
					ones_blocked += 1
				if b != current and b != self.empty and a == self.empty:
					ones_blocked += 1
			if twos_created > 0:
				features[space][5] = 1.0
			if twos_created > 1:
				features[space][6] = 1.0
			if ones_blocked > 1:
				features[space][7] = 1.0
		return features

# Infinite loop of play against an agent.
def interactive_play(g, agent, print_features=False, print_policy=True, print_value=True):
	game = 0
	while True:
		g.reset()
		a_to_move = game % 2 == 0
		print 'New game!', 'Agent moves first.' if a_to_move else 'You move first.'
		while not g.is_game_over():
			move = agent.select_move(print_debug=True) if a_to_move else raw_input('Select a move:')
			try:
				move = int(move)
			except:
				if move in ['q', 'quit']:
					return
				if move in ['r', 'resign']:
					g.winner = 'resignation'
					a_to_move = False
					break
				print 'Please input a valid legal index.'
				continue
			if move in g.legal_moves():
				g.make_move(move)
				a_to_move = not a_to_move
				g.print_board()
				if print_features:
					print g.make_features()
				print
		game += 1
		if g.winner == '_':
			print 'Game Drawn'
		else:
			if a_to_move:
				print 'You win!'
			else:
				print agent.name, 'wins!'

# Runs a match between two agents.
def play_match(g, 
			   agent_a, 
			   agent_b, 
			   num_games, 
			   print_games=False, 
			   print_results=True, 
			   print_a_losses=False, 
			   evaluator=None):
	results = {agent_a.name : 0, 
			   agent_b.name : 0, 
			   "Draws" : 0}
	results_by_player = { 
			   agent_a.name+'-O':0, 
			   agent_a.name+'-X':0,
			   agent_b.name+'-O':0, 
			   agent_b.name+'-X':0}
	if evaluator is not None:
		optimality = {agent_a.name+'_optimal': 0,
					  agent_a.name+'_suboptimal': 0, 
					  agent_b.name+'_optimal': 0, 
					  agent_b.name+'_suboptimal': 0}
 	for game in range(num_games):
		g.reset()
		a_to_move = game % 2 == 0
		if print_games:
			x = agent_a.name if a_to_move else agent_b.name
			o = agent_b.name if a_to_move else agent_a.name
			print x, 'is playing as X - ', o, 'is playing as O'
		moves = []
		while not g.is_game_over():
			move = agent_a.select_move() if a_to_move else agent_b.select_move()
			if evaluator is not None:
				optimal_moves = evaluator.optimal_moves()
				key_prefix = agent_a.name if a_to_move else agent_b.name
				if move in optimal_moves:
					optimality[key_prefix+'_optimal'] += 1
				else:
					optimality[key_prefix+'_suboptimal'] += 1
			moves.append(move)
			g.make_move(move)
			a_to_move = not a_to_move
			if print_games:
				g.print_board()
				print
		if g.winner == '_':
			results["Draws"] += 1
		else:
			if a_to_move:
				winning_agent = agent_b.name
				if print_a_losses:
					print agent_a.name, 'lost! Game:'
					print moves
			else:
				winning_agent = agent_a.name
			results[winning_agent] += 1
			results_by_player[winning_agent+'-'+g.winner] +=1
		if print_games:
			print "Draw!" if g.winner == '_' else winning_agent + ' as ' + g.winner + ' WINS'
	overall_results = sorted(results.items())
	results_breakdown = sorted(results_by_player.items())
	if print_results:
		print 'Results for', num_games, 'game match between', agent_a.name, 'and', agent_b.name + ':'
		print overall_results
		print results_breakdown
		if evaluator is not None:
			print optimality
	return overall_results, results_breakdown
