from __future__ import division

import itertools
import numpy as np

class TicTacToe(object):
	def __init__(self, board_size, num_in_a_row):
		self.value_mapping = {'_':0, 'X':1.0, 'O':-1.0}
		self.empty = '_'
		self.winner = self.empty
		self.players = ['X', 'O']
		self.first_player_to_move = True
		self.board_size = board_size
		self.num_in_a_row = num_in_a_row
		self.board = [[[self.empty] for i in range(board_size)] for j in range(board_size)]
		self.columns = [[self.board[j][i] for j in range(board_size)] for i in range(board_size)]
		self.diagonals = [[] for i in range((board_size*4)-2)]
		for i in range(board_size):
			for j in range(board_size):
				self.diagonals[i+j].append(self.board[i][j])
				self.diagonals[(board_size-i-1)+j+(2*board_size)-1].append(self.board[i][j])
		self.moves = []

	def is_game_over(self):
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
		return len(self.legal_moves()) == 0


	def print_board(self):
		for row in self.board:
			print '|' + '|'.join([elt[0] for elt in row]) + '|'

	def legal_moves(self):
		return [i for i in range(self.board_size**2) 
			if self.board[i//self.board_size][i%self.board_size][0] == self.empty]

	def make_move(self, move):
		self.board[move//self.board_size][move%self.board_size][0] = 'X' if self.first_player_to_move else 'O'
		self.moves.append(move)
		self.first_player_to_move = not self.first_player_to_move

	def undo_move(self):
		move = self.moves.pop()
		self.winner = self.board[move//self.board_size][move%self.board_size][0] = self.empty
		self.first_player_to_move = not self.first_player_to_move

	def reset(self):
		self.winner = self.empty
		self.first_player_to_move = True
		for i in range(self.board_size):
			for j in range(self.board_size):
				self.board[i][j][0] = self.empty

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

def interactive_play(g, agent):
	game = 0
	while True:
		g.reset()
		a_to_move = game % 2 == 0
		print 'New game!', 'Agent moves first.' if a_to_move else 'You move first.'
		while not g.is_game_over():
			move = agent.select_move(print_policy=True) if a_to_move else raw_input('Select a move:')
			try:
				move = int(move)
			except:
				print 'Please input a valid legal index.'
				continue
			if move in g.legal_moves():
				g.make_move(move)
				a_to_move = not a_to_move
				g.print_board()
				print
		game += 1
		if g.winner == '_':
			print 'Game Drawn'
		else:
			if a_to_move:
				print 'You win!'
			else:
				print agent.name, 'wins!'

def play_match(g, agent_a, agent_b, num_games, print_games=False, print_results=True):
	results = {agent_a.name : 0, 
			   agent_b.name : 0, 
			   "Draws" : 0}
	results_by_player = { 
			   agent_a.name+'-O':0, 
			   agent_a.name+'-X':0,
			   agent_b.name+'-O':0, 
			   agent_b.name+'-X':0}
 	for game in range(num_games):
		g.reset()
		a_to_move = game % 2 == 0
		if print_games:
			x = agent_a.name if a_to_move else agent_b.name
			o = agent_b.name if a_to_move else agent_a.name
			print x, 'is playing as X - ', o, 'is playing as O'
		while not g.is_game_over():
			move = agent_a.select_move() if a_to_move else agent_b.select_move()
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
	return overall_results, results_breakdown