
class RLAgent(object):
	def __init__(self, name, game, model):
		self.name = name
		self.game = game
		self.model = model

	def select_move(self, print_policy=False):
		legal_moves = self.game.legal_moves()
		move_vals = self.model.policy(self.game.transform_state())
		if print_policy:
			self.game.print_board()
			print 'Policy Probabilities', move_vals
		best_move = -1
		for move in legal_moves:
			if best_move == -1 or move_vals[move] > move_vals[best_move]:
				best_move = move
		return best_move

	def generate_data(self, num_games, num_rollouts, rollout_depth, t, c_puct, opponent=None):
		data = []
		for sim in range(num_games):
			if sim % 10 == 0:
				print "Starting to simulate game", sim+1, "of", num_games
			self.game.reset()
			tree = GameNode(None, self.game, self.model, t, c_puct)
			while not self.game.is_game_over():
				for rollout in range(num_rollouts):
					tree.rollout(rollout_depth, rollout_depth)
				moves_and_probs = tree.get_probabilities().items()
				selected_move = np.random.choice([x[0] for x in moves_and_probs], p=[x[1] for x in moves_and_probs])
				self.game.make_move(selected_move)
				tree = tree.children[selected_move]
			outcome = self.game.value_mapping[self.game.winner]
			tree = tree.parent
			while tree is not None:
				self.game.undo_move()
				# Set the illegal move probs to -1.
				probs = -1*np.ones([self.game.board_size**2])
				legal_probs = tree.get_probabilities()
				for move in legal_probs:
					probs[move] = legal_probs[move]
				data.append((self.game.transform_state(), outcome, probs))
				tree = tree.parent
		return data

	def train(self, num_epochs, games_per_epoch, rollouts_per_move, rollout_depth):
		random = agents.RandomAgent('Random', self.game)
		dumb = agents.DumbAgent('Dumb', self.game)
		minimax = agents.MinimaxAgent('Minimax', self.game)
		starting_t = 2
		final_t = 0.5
		starting_c_puct = 3.0
		final_c_puct = 0.5
		for epoch in range(num_epochs):
			print "Epoch", epoch+1, "of", num_epochs
			t = starting_t - (1+epoch)*((starting_t - final_t)/num_epochs)
			c_puct = starting_c_puct - (1+epoch)*((starting_c_puct - final_c_puct)/num_epochs)
			print 't is', t, ', c_puct is', c_puct
			play_match(self.game, self, random, 1000)
			play_match(self.game, self, dumb, 4)
			play_match(self.game, self, minimax, 4)
			play_match(self.game, self, self, 100)
			data = self.generate_data(games_per_epoch, rollouts_per_move, rollout_depth, t, c_puct)
			np.random.shuffle(data)
			self.model.train(data, games_per_epoch)