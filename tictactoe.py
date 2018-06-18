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

	# Convenience function to transform the board state into a stack of layers.
	# The first layer is the locations of X, the second O, and the third is entirely
	# 1.0 if it's O-to-move and 0.0 if it's X-to-move.
	def state(self):
		stack = np.zeros([self.board_size, self.board_size, 3])
		if not self.first_player_to_move:
			stack[2, :, :] = np.ones([self.board_size, self.board_size])
		for i in range(self.board_size):
			for j in range(self.board_size):
				if self.board[i][j][0] == 'X':
					stack[0][i][j] = 1.0
				elif self.board[i][j][0] == 'O':
					stack[1][i][j] = 1.0
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

# Infinite loop of play against an agent.
def interactive_play(g, agent, print_policy=True, print_value=True):
	game = 0
	while True:
		g.reset()
		a_to_move = game % 2 == 0
		print 'New game!', 'Agent moves first.' if a_to_move else 'You move first.'
		while g.result() is None:
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
			if move in g.available_actions():
				g.take_action(move)
				a_to_move = not a_to_move
				g.print_state()
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
		while g.result() is None:
			move = agent_a.select_move() if a_to_move else agent_b.select_move()
			if evaluator is not None:
				optimal_moves = evaluator.optimal_moves()
				key_prefix = agent_a.name if a_to_move else agent_b.name
				if move in optimal_moves:
					optimality[key_prefix+'_optimal'] += 1
				else:
					optimality[key_prefix+'_suboptimal'] += 1
			moves.append(move)
			g.take_action(move)
			a_to_move = not a_to_move
			if print_games:
				g.print_state()
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
