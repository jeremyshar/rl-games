from __future__ import division

import tensorflow as tf
import numpy as np

import agents as agent_lib
import game as game_lib
import rl_agent

class GameNode(object):
	def __init__(self, parent, game, model, t, c_puct, opponent=None, opponent_to_move=False):
		self.parent = parent
		self.visit_count = 0
		self.average_outcome = 0
		self.game = game
		self.model = model
		self.children = {i:None for i in self.game.legal_moves()}
		self.expanded = False
		self.t = t
		self.c_puct = c_puct
		self.opponent = opponent
		self.opponent_to_move = opponent_to_move


	def expand(self):
		if not self.expanded:
			self.expanded = True
			for child in self.children:
				self.game.make_move(child)
				self.children[child] = GameNode(self, self.game, self.model, self.t, self.c_puct, self.opponent, not self.opponent_to_move)
				self.game.undo_move()

	def backup(self, outcome, remaining_climb):
		if self.parent is not None and remaining_climb > 0:
			self.visit_count += 1
			self.average_outcome += (outcome - self.average_outcome)/self.visit_count
			self.game.undo_move()
			self.parent.backup(outcome, remaining_climb-1)

	def rollout(self, rollout_depth, remaining_depth, use_features):
		if remaining_depth == 0 or self.game.is_game_over():
			if self.game.is_game_over():
				outcome = self.game.value_mapping[self.game.winner]
			else:
				outcome = self.model.value(self.game.transform_state())
			self.backup(outcome, rollout_depth-remaining_depth)
		else:
			self.expand()
			multiplier = 1.0 if self.game.first_player_to_move else -1.0
			p_values = self.model.policy(self.game.transform_state() if not use_features else self.game.make_features())
			stats = [(i, self.children[i].average_outcome, self.children[i].visit_count, p_values[i]) for i in self.children]
			total_visits = sum([x[2] for x in stats])
			scores = [(multiplier*x[1])+(self.c_puct*np.sqrt(total_visits)*x[3]/(1+x[2])) for x in stats]
			selected_move = stats[np.argmax(scores)][0]
			if self.opponent is not None and self.opponent_to_move:
				self.selected_move = self.opponent.select_move()
			self.game.make_move(selected_move)
			self.children[selected_move].rollout(rollout_depth, remaining_depth-1, use_features)

	def get_probabilities(self, print_probabilities=False):
		counts = [(child, self.children[child].visit_count) for child in self.children]
		numerators = np.power([x[1] for x in counts], 1/self.t)
		denominator = np.sum(numerators)
		probs = np.divide(numerators, denominator)
		prob_dict = {counts[i][0]:probs[i] for i in range(len(probs))}
		if print_probabilities:
			for c in sorted(self.children):
				print 'Move:', c, 'Visit Count:', self.children[c].visit_count, 'Average Outcome:', self.children[c].average_outcome, 'Prob:', prob_dict[c]
		return prob_dict

class NeuralNetModel(object):
	def __init__(self, game, sess):
		self.game = game
		self.sess = sess
		self.state_op, self.value_op, self.policy_op, self.target_value, self.target_policy, self.train_op, self.loss_op = self.build_net()
		self.sess.run(tf.global_variables_initializer())

	def build_net(self):
		batch = tf.placeholder(dtype=tf.float32, shape=[None, self.game.board_size, self.game.board_size, 3])
		flat_batch = tf.reshape(batch, [-1, self.game.board_size*self.game.board_size*3])
		first_layer = tf.layers.dense(inputs=flat_batch, units=self.game.board_size*self.game.board_size*3, activation=tf.nn.relu)
		#second_layer = tf.layers.dense(inputs=first_layer, units=self.game.board_size*self.game.board_size, activation=tf.nn.relu)
		#third_layer = tf.layers.dense(inputs=second_layer, units=self.game.board_size*self.game.board_size, activation=tf.nn.relu)
		last_layer = first_layer
		policy = tf.nn.softmax(tf.squeeze(tf.layers.dense(inputs=last_layer, units=self.game.board_size*self.game.board_size, activation=tf.nn.relu)))
		value = tf.squeeze(tf.layers.dense(inputs=last_layer, units=1, activation=tf.nn.relu))
		target_values = tf.placeholder(dtype=tf.float32, shape=[None])
		target_policies = tf.placeholder(dtype=tf.float32, shape=[None, self.game.board_size**2])
		loss = (0*tf.reduce_sum(tf.square(target_values-value)) - 
			   	tf.reduce_sum(tf.where(tf.greater(target_policies, -1), 
			   						   tf.multiply(target_policies, tf.log(policy+1e-3)), 
			   						   tf.zeros_like(policy))))
		train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
		return batch, value, policy, target_values, target_policies, train_op, loss

	def policy(self, state):
		pol = self.sess.run(self.policy_op, feed_dict={self.state_op: np.expand_dims(state, 0)})
		return pol

	def value(self, state):
		return self.sess.run(self.value_op, feed_dict={self.state_op: np.expand_dims(state, 0)})

	def train(self, data, games_per_epoch):
		state_batches = np.array_split(np.array([x[0] for x in data]), games_per_epoch)
		value_batches = np.array_split(np.array([x[1] for x in data]), games_per_epoch)
		policy_batches = np.array_split(np.array([x[2] for x in data]), games_per_epoch)
		total_loss = 0
		for i in range(len(state_batches)):
			for j in range(50):
				loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict={self.state_op: state_batches[i], 
																	  			  self.target_value: value_batches[i],
																	  			  self.target_policy: policy_batches[i]})
				total_loss += loss
		print "Total loss =", total_loss

class ConvolutionalModel(object):

	def __init__(self, game, sess):
		self.game = game
		self.sess = sess
		self.state_op, self.value_op, self.policy_op, self.target_value, self.target_policy, self.train_op, self.loss_op = self.build_net()
		self.sess.run(tf.global_variables_initializer())

	def build_net(self):
		batch = tf.placeholder(dtype=tf.float32, 
							   shape=[None, self.game.board_size, self.game.board_size, 3])
		for layer in range(int(np.ceil(np.sqrt(self.game.board_size)))):
			if layer == 0:
				conv = tf.layers.conv2d(inputs=batch, 
							 			filters=self.game.board_size**2, 
							 			kernel_size=[self.game.num_in_a_row, self.game.num_in_a_row], 
							 			padding='same',
							 			activation=tf.nn.relu)
			else:
				kernel_size = min(self.game.board_size, layer + self.game.num_in_a_row)
				conv = tf.layers.conv2d(inputs=conv,
									    filters=max(1, self.game.board_size**2//layer),
									    kernel_size=[kernel_size, kernel_size],
									    padding='same',
									    activation=tf.nn.relu)
		reshaped = tf.contrib.layers.flatten(conv)
		policy = tf.nn.softmax(tf.squeeze(tf.layers.dense(inputs=reshaped,
								 			units=self.game.board_size**2, 
								 			activation=tf.nn.relu)))
		value = tf.squeeze(tf.layers.dense(inputs=reshaped, units=1, activation=tf.nn.relu))
		target_values = tf.placeholder(dtype=tf.float32, shape=[None])
		target_policies = tf.placeholder(dtype=tf.float32, shape=[None, self.game.board_size**2])
		loss = (tf.reduce_sum(tf.square(target_values-value)) - 
			   	tf.reduce_sum(tf.where(tf.greater(target_policies, -1), 
			   						   tf.multiply(target_policies, tf.log(policy)), 
			   						   tf.zeros_like(policy))))
		train_op = tf.train.AdamOptimizer(1.0).minimize(loss)
		return batch, value, policy, target_values, target_policies, train_op, loss


	def policy(self, state):
		return self.sess.run(self.policy_op, feed_dict={self.state_op: np.expand_dims(state, 0)})

	def value(self, state):
		return self.sess.run(self.value_op, feed_dict={self.state_op: np.expand_dims(state, 0)})

	def train(self, data, games_per_epoch):
		state_batches = np.array_split(np.array([x[0] for x in data]), games_per_epoch)
		value_batches = np.array_split(np.array([x[1] for x in data]), games_per_epoch)
		policy_batches = np.array_split(np.array([x[2] for x in data]), games_per_epoch)
		total_loss = 0
		for i in range(len(state_batches)):
			for j in range(50):
				loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict={self.state_op: state_batches[i], 
																	  			  self.target_value: value_batches[i],
																	  			  self.target_policy: policy_batches[i]})
				total_loss += loss
		print "Total loss =", total_loss

class FeatureModel(object):

	def __init__(self, game, sess):
		self.game = game
		self.sess = sess
		self.train_op, self.policy_op, self.target_policy, self.state_op, self.weights, self.biases = self.build_model()
		self.sess.run(tf.global_variables_initializer())

	def build_model(self):
		batch = tf.placeholder(dtype=tf.float32, shape=[None, 9, 8])
		target_policies = tf.placeholder(dtype=tf.float32, shape=[None, 9])
		weights = tf.get_variable('weights', [8], tf.float32)
		biases = tf.get_variable('biases', [9], tf.float32)
		predicted_policies = tf.nn.softmax(tf.squeeze(tf.reduce_sum(batch*weights, axis=2) + biases))
		loss = tf.reduce_sum(tf.where(tf.greater(target_policies, -1), tf.square(target_policies-predicted_policies), tf.zeros_like(target_policies)))
		train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
		return train_op, predicted_policies, target_policies, batch, weights, biases

	def policy(self, state):
		y = self.sess.run(self.policy_op, feed_dict={self.state_op: np.expand_dims(state, 0)})
		return y

	def train(self, data, games_per_epoch):
		state_batches = np.array_split(np.array([x[0] for x in data]), games_per_epoch)
		value_batches = np.array_split(np.array([x[1] for x in data]), games_per_epoch)
		policy_batches = np.array_split(np.array([x[2] for x in data]), games_per_epoch)
		total_loss = 0
		for j in range(5):
			for i in range(len(state_batches)):
				self.sess.run(self.train_op, feed_dict={self.state_op: state_batches[i], 
													   self.target_policy: policy_batches[i]})


class TableModel(object):

	def __init__(self):
		self.table = {}

	def lookup(self, state, delete_entry=False):
		for flip in [False, True]:
			for rotation_amt in range(3):
				rotated = np.rot90(state, k=rotation_amt, axes=(2, 1))
				symmetry = np.fliplr(rotated) if flip else rotated
				if symmetry.tostring() in self.table:
					stats = self.table[symmetry.tostring()]
					rotated_moves = np.rot90(np.reshape(stats[2], [3, 3]), k=rotation_amt, axes=(1, 0))
					flipped_moves = np.fliplr(rotated_moves) if flip else rotated_moves
					if delete_entry:
						del self.table[symmetry.tostring()]
					return [stats[0], stats[1], np.reshape(flipped_moves, [-1])]
		return None


	def policy(self, state):
		res = self.lookup(state)
		if res is not None:
			return res[2]
		else:
			return [1/9]*9

	def value(self, state):
		res = self.lookup(state)
		if res is not None:
			return res[1]
		else:
			return 0

	def train(self, data, games_per_epoch):
		for datum in data:
			stats = self.lookup(datum[0], delete_entry=True)
			if stats is None:
				self.table[datum[0].tostring()] = [1, datum[1], datum[2]]
			else:
				stats[0] += 1
				stats[1] += (datum[1] - stats[1])/stats[0]
				for i in range(len(datum[2])):
					stats[2][i] += (datum[2][i] - stats[2][i])/stats[0]
				self.table[datum[0].tostring()] = stats


class RLAgent(object):
	def __init__(self, name, game, model, is_feature_model=False):
		self.name = name
		self.game = game
		self.model = model
		self.use_features = is_feature_model


	def select_move(self, print_policy=False):
		legal_moves = self.game.legal_moves()
		move_vals = self.model.policy(self.game.transform_state() if not self.use_features else self.game.make_features())
		if print_policy:
			self.game.print_board()
			print 'Policy Probabilities', move_vals
		best_move = -1
		for move in legal_moves:
			if best_move == -1 or move_vals[move] > move_vals[best_move]:
				best_move = move
		return best_move

	def generate_data(self, num_games, num_rollouts, rollout_depth, t, c_puct, opponent=None, use_opponent_for_mcts=True):
		data = []
		for sim in range(num_games):
			if sim % (num_games // 10) == 0:
				print "Starting to simulate game", sim+1, "of", num_games
			self.game.reset()
			opponent_to_move = sim % 2 == 0
			tree = GameNode(None, self.game, self.model, t, c_puct, opponent if use_opponent_for_mcts else None, opponent_to_move)
			while not self.game.is_game_over():
				if opponent is not None and opponent_to_move:
					selected_move = opponent.select_move()
					tree.expand()
				else:
					for _ in range(num_rollouts):
						tree.rollout(rollout_depth, rollout_depth, self.use_features)
					moves_and_probs = tree.get_probabilities().items()
					selected_move = np.random.choice([x[0] for x in moves_and_probs], p=[x[1] for x in moves_and_probs])
				opponent_to_move = not opponent_to_move
				self.game.make_move(selected_move)
				tree = tree.children[selected_move]
			outcome = self.game.value_mapping[self.game.winner]
			tree = tree.parent
			while tree is not None:
				self.game.undo_move()
				opponent_to_move = not opponent_to_move
				# Set the illegal move probs to -1.
				probs = -1*np.ones([self.game.board_size**2])
				legal_probs = tree.get_probabilities()
				for move in legal_probs:
					probs[move] = legal_probs[move]
				tree = tree.parent
				if opponent is not None and opponent_to_move:
					continue
				data.append((self.game.transform_state() if not self.use_features else self.game.make_features(), outcome, probs))
		return data

	def train(self, num_epochs, games_per_epoch, rollouts_per_move, rollout_depth):
		random = agent_lib.RandomAgent('Random', self.game)
		dumb = agent_lib.DumbAgent('Dumb', self.game)
		minimax = agent_lib.MinimaxAgent('Minimax', self.game)
		starting_t = 2
		final_t = 0.5
		starting_c_puct = 3.0
		final_c_puct = 0.5
		for epoch in range(num_epochs):
			print "Epoch", epoch+1, "of", num_epochs
			t = starting_t - (1+epoch)*((starting_t - final_t)/num_epochs)
			c_puct = starting_c_puct - (1+epoch)*((starting_c_puct - final_c_puct)/num_epochs)
			print 't is', t, ', c_puct is', c_puct
			game_lib.play_match(self.game, self, random, 1000)
			game_lib.play_match(self.game, self, dumb, 2)
			game_lib.play_match(self.game, self, minimax, 2)
			game_lib.play_match(self.game, self, self, 2)
			data = self.generate_data(games_per_epoch, rollouts_per_move, rollout_depth, t, c_puct, opponent=None, use_opponent_for_mcts=False)
			np.random.shuffle(data)
			self.model.train(data, games_per_epoch)
			if self.use_features:
				print self.model.sess.run([self.model.weights, self.model.biases])
			

def main():
	with tf.Graph().as_default():
		with tf.Session() as sess:
			g = game_lib.TicTacToe(3, 3)
			a = agent_lib.RandomAgent('Random', g)
			b = agent_lib.DumbAgent('Dumb', g)
			c = agent_lib.MinimaxAgent('Minimax', g)
			rl = RLAgent('RL', g, FeatureModel(g, sess), is_feature_model=True)
			rl.train(10, 300, 30, 9)
			print "Match with RandomAgent"
			game_lib.play_match(g, rl, a, 1000, False)
			print "Match with DumbAgent"
			game_lib.play_match(g, rl, b, 2, False)
			print "Match with MinimaxAgent"
			game_lib.play_match(g, rl, c, 2, False)
			print "Match with self"
			game_lib.play_match(g, rl, rl, 2, False)
			print "Match with you!"
			game_lib.interactive_play(g, rl)

main()