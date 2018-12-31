import collections
import numpy as np

import abstract
import agents as ag
import game_lib as gl
import game_tree as gt

Config = collections.namedtuple('Config', ['training_epochs', # An integer, the number of epochs to train for.
										   'games_per_epoch', # An integer, the number of games to simulate per epoch.
										   'rollouts_per_move', # An integer, the number of rollouts to perform each move.
										   'rollout_depth', # An integer, the maximum depth for each rollout.
										   'rollout_policy', # Functions taking a dictionary of action indices->Actions and select an Action.
										   'play_policy',
										   'inference_policy',
										   'opponent_rollout_policy', # Functions taking no arguments that select a legal move to play (or None for self-play).
										   'opponent_play_policy', 
										   'policy_target', # A function taking a dictionary of moves->Actions and outputting the learning target for the policy
										   'inference_rollouts_per_move', # An integer, how many rollouts to do when selecting moves in matches (0 to use the raw policy).
										   'inference_rollout_depth' # An integer, how deep the inference rollouts should be (0 to use the raw policy)
										   ])


class RLAgent(abstract.Agent):
	def __init__(self, name, game, model, config, match_opponents=(), games_per_match=0, evaluator=None):
		self.name = name
		self.game = game
		self.model = model
		self.config = config
		self.match_opponents = match_opponents
		self.games_per_match = games_per_match
		self.evaluator = evaluator

	def transition(self, states, selected_action):
		"""Helper function for transitioning between states in a simulated game.

		Args:
			states: A map from state keys to visited State objects.
			selected_action: The action to take.

		Returns:
			The new current state.
		"""
		self.game.take_action(self.game.index_to_action(selected_action.action if hasattr(selected_action, 'action') else selected_action))
		new_state_key = self.game.state_key()
		if new_state_key not in states:
			return gt.State(states=states,
							game=self.game,
							model=self.model,
							rollout_policy=self.config.rollout_policy,
							opponent_rollout_policy=self.config.opponent_rollout_policy)
		return states[new_state_key]

	def simulate_game(self):
		"""Simulates a single game, returning a list of (state, target value, target policy) tuples."""
		self.game.reset()
		states = {}
		# If there's an opponent, they should start half the time.
		if self.config.opponent_play_policy and np.random.choice([True, False]):
			root = self.transition(states, self.config.opponent_play_policy())
		else:
			root = gt.State(states=states, 
					 		game=self.game, 
				 			model=self.model,
				 			rollout_policy=self.config.rollout_policy,
				 			opponent_rollout_policy=self.config.opponent_rollout_policy)
		game_states = []
		while not root.is_terminal:
			game_states.append(root)
			for _ in xrange(self.config.rollouts_per_move):
				root.rollout(rollout_depth=self.config.rollout_depth, current_depth=0, rollout_actions=[])
			selected_action = self.config.play_policy(root.actions)
			root = self.transition(states, selected_action)
			if self.config.opponent_play_policy and not root.is_terminal:
				selected_action = self.config.opponent_play_policy()
				root = self.transition(states, selected_action)
		return zip([s.id for s in game_states], [root.terminal_value]*len(game_states), [self.config.policy_target(state.actions) for state in game_states])

	def simulate_epoch(self, print_progress=False):
		"""Returns the data collected from an epoch of game simulations."""
		data = []
		for game_number in xrange(self.config.games_per_epoch):
			if print_progress and game_number % 10 == 1:
				print 'Simulating game #{}'.format(game_number)
			data.extend(self.simulate_game())
		return data

	def train(self, print_progress=False, save_data=False, load_data=False):
		"""Trains the agent according to its config, updating its model."""
		for e in xrange(self.config.training_epochs):
			if load_data:
				data = self.load_epoch('data/resampled_epoch_{}.txt'.format(e + 1))
			else:
				data = self.simulate_epoch(print_progress=print_progress)
			if save_data and not load_data:
				self.save_epoch('data/ml_epoch_{}.txt'.format(e), data)
			self.model.train(data)
			for opponent in self.match_opponents:
				gl.play_match(self.game, 
							  self, 
							  opponent, 
							  num_games=self.games_per_match,
							  evaluator=self.evaluator)

	def select_move(self):
		"""Selects the move to play at inference time.

		Returns:
			The action key of the action to be taken.
		"""
		states = {}
		root = gt.State(states=states, 
				 		game=self.game, 
			 			model=self.model,
			 			rollout_policy=self.config.rollout_policy,
			 			opponent_rollout_policy=self.config.opponent_rollout_policy)
		for _ in xrange(self.config.inference_rollouts_per_move):
			root.rollout(self.config.inference_rollout_depth, current_depth=0, rollout_actions=[])
		return self.game.index_to_action(self.config.inference_policy(root.actions).action)

	def save_epoch(self, filename, data):
		with open(filename, 'w') as f:
			for position, value, policy in data:
				serialized_position = ','.join(['{:0.5f}' for i in range(27)]).format(*np.reshape(position, [-1]))
				serialized_value = '{:0.5f}'.format(value)
				serialized_policy = ','.join(['{:0.5f}' for i in range(9)]).format(*policy)
				line = '{}|{}|{}\n'.format(serialized_position, serialized_value, serialized_policy)
				f.write(line)

	def load_epoch(self, filename):
		def format_line(line):
			pos, val, pol = line.split('|')
			return (np.reshape(map(float, pos.split(',')), (3, 3, 3)), float(val), np.array(map(float, pol.split(','))))
		with open(filename, 'r') as f:
			return [format_line(line) for line in f.read().splitlines()]




	