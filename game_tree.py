from __future__ import division
from recordclass import recordclass

Action = recordclass('Action', ['before', # The state the action was taken in.
								'after', # The states the action leads to.
								'action', # The id of the action.
								'visit_count', # The number of times the action has been visited.
								'average_value', # The average value of the action.
								'prior', # The prior value for the action.
								'for_maximizer']) # A boolean, True if this action is for the maximizer
class State(object):
	"""An object representing a state in the game tree."""
	def __init__(self, 
				 states, 
				 game, 
				 model,
				 rollout_policy,
				 opponent_rollout_policy):
		"""Constructor for a State.

		Args:
			states: A global map from state keys to all visited states in the game tree.
			game: The shared Game object.
			model: The shared Model object.
			rollout_policy: A function with signature Dict[int, Action]->int used to select
				the action key of the action to take during rollouts.
			opponent_rollout_policy: A function with signature Dict[int, Action]->int used to 
				the action key of the action to take for the opponent during rollouts. None
				means self-play (the rollout policy is used for both players).  
		"""
		self.game = game # Needs access to the game so that it can use game dynamics
		self.model = model # Needs access to the model to evaluate future states.
		self.states = states # A map from string ids to all the known states so that we don't create duplicate States.
		self.id = game.state() # A numpy array that uniquely identifies this state.
		self.prior = model.value(self.id) # The model's score of the state.
		self.states[game.state_key()] = self # Add this state to the global list of states
		self.rollout_policy = rollout_policy # Function that governs how to select a move during rollouts.
		self.opponent_rollout_policy = opponent_rollout_policy # Function that governs how the opponent selects a move during rollouts (or None for self play).
		self.terminal_value = game.result()
		self.is_terminal = self.terminal_value is not None
		self.maximizer_to_act = game.maximizer_to_act()
		self.actions = {}
		if not self.is_terminal:
			action_priors = model.policy(self.id)
			self.actions = {self.game.action_index(move): Action(before=self, 
									 	 after={},
									 	 action=self.game.action_index(move), 
									     visit_count=0, 
									     average_value=0,
									     prior=action_priors[self.game.action_index(move)], 
									     for_maximizer=self.maximizer_to_act) for move in game.available_actions()}


	def backup(self, outcome, current_depth, rollout_actions):
		"""Backs up the results from the bottom of a rollout, updating 
		statistics and undoing the moves.

		Args:
			outcome: A float, outcome to propagate back through the game tree
				for each action's average_value property.
			current_depth: An int, the current depth of the backup in the game tree
				i.e. how much farther the backup should go.
			rollout_actions: A list of the Actions taken during the rollout_policy
				so that the backup can be properly applied.
		"""
		if current_depth != 0:
			action = rollout_actions[current_depth-1]
			if action is None:
				self.game.undo_action()
				current_depth -= 1
				action = rollout_actions[current_depth-1]
			action.visit_count += 1
			action.average_value += (outcome - action.average_value)/action.visit_count
			self.game.undo_action()
			action.before.backup(outcome, current_depth-1, rollout_actions)

	
	def rollout(self, rollout_depth, current_depth, rollout_actions):
		"""Performs a rollout to the specified depth (or until the game ends) 
		and backs up the results to all actions taken during the rollout.

		Args:
			rollout_depth: An int, the maximum depth the rollout will go to.
			current_depth: An int, the current depth of the rollout.
			rollout_actions: A list of the Actions taken in the rollout so far.
		"""
		if rollout_depth == current_depth or self.is_terminal or (rollout_depth == current_depth + 1 and self.opponent_rollout_policy):
			if self.is_terminal:
				outcome = self.terminal_value
			else:
				outcome = self.prior
			self.backup(outcome, current_depth, rollout_actions)
		else:
			selected_action = self.rollout_policy(self.actions)
			self.game.take_action(self.game.index_to_action(selected_action.action))
			rollout_actions.append(selected_action)
			if self.opponent_rollout_policy and self.game.result() is None:
				opponent_action = self.opponent_rollout_policy()
				if hasattr(opponent_action, 'action'):
					opponent_action = opponent_action.action
				self.game.take_action(self.game.index_to_action(opponent_action))
				current_depth += 1
				# Append None to signify an opponent move.
				rollout_actions.append(None)
			new_state_id = self.game.state_key()
			if new_state_id in self.states:
				selected_action.after[new_state_id] = self.states[new_state_id]
			else:
				selected_action.after[new_state_id] = State(states=self.states,
															game=self.game, 
				 											model=self.model,
				 											rollout_policy=self.rollout_policy,
				 											opponent_rollout_policy=self.opponent_rollout_policy)
			selected_action.after[new_state_id].rollout(rollout_depth, current_depth+1, rollout_actions)

