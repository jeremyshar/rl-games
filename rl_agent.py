import collections
import numpy as np

import abstract
import agents as ag
import game_lib as gl
import game_tree as gt

Config = collections.namedtuple(
    'Config',
    [
        # An int, the number of epochs to train for.
        'training_epochs',
        # An int, the number of games to simulate per epoch.
        'games_per_epoch',
        # An int, the number of rollouts to perform each move.
        'rollouts_per_move',
        # An int, the maximum depth for each rollout.
        'rollout_depth',
        # Functions taking a dictionary of action indices->Actions and returning
        # an Action.
        'rollout_policy',
        'play_policy',
        'inference_policy',
        # Functions taking no arguments that select a legal move to play. Should
        # be set to None for self-play.
        'opponent_rollout_policy',
        'opponent_play_policy',
        # A function taking a dictionary of moves->Actions and outputting the
        # learning target for the policy
        'policy_target',
        # An int, the number of rollouts to do when selecting moves at inference
        # time (0 uses the raw policy).
        'inference_rollouts_per_move',
        # An int, the depth of the inference rollouts (0 uses the raw policy).
        'inference_rollout_depth'
    ])


class RLAgent(abstract.Agent):
  """An agent that learns to play a game with Reinforcement Learning."""

  def __init__(self,
               name,
               game,
               model,
               config,
               match_opponents=(),
               games_per_match=0,
               evaluator=None):
    """Constructor for the RLAgent.

    Args:
      name: The name of the agent (for display purposes).
      game: The game that the agent is learning to play.
      model: The model the agent uses as its policy and value functions.
      config: The config for how to train and play.
      match_opponents: A list of opponents to play in intermediate matches.
      games_per_match: The number of games to play in each intermediate match.
      evaluator: Either None (for no evaluation) or an agent with an
        'optimal_moves' method that evaluates the match games.
    """
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
    if hasattr(selected_action, 'action'):
      converted = selected_action.action
    else:
      converted = selected_action
    self.game.take_action(self.game.index_to_action(converted))
    new_state_key = self.game.state_key()
    if new_state_key not in states:
      return gt.State(
          states=states,
          game=self.game,
          model=self.model,
          rollout_policy=self.config.rollout_policy,
          opponent_rollout_policy=self.config.opponent_rollout_policy)
    return states[new_state_key]

  def simulate_game(self):
    """Simulates a single game.

    Returns: A list of (state, target value, target policy) tuples.
    """
    self.game.reset()
    states = {}
    # If there's an opponent, they should start half the time.
    if self.config.opponent_play_policy and np.random.choice([True, False]):
      root = self.transition(states, self.config.opponent_play_policy())
    else:
      root = gt.State(
          states=states,
          game=self.game,
          model=self.model,
          rollout_policy=self.config.rollout_policy,
          opponent_rollout_policy=self.config.opponent_rollout_policy)
    game_states = []
    while not root.is_terminal:
      game_states.append(root)
      for _ in xrange(self.config.rollouts_per_move):
        root.rollout(
            rollout_depth=self.config.rollout_depth,
            current_depth=0,
            rollout_actions=[])
      selected_action = self.config.play_policy(root.actions)
      root = self.transition(states, selected_action)
      if self.config.opponent_play_policy and not root.is_terminal:
        selected_action = self.config.opponent_play_policy()
        root = self.transition(states, selected_action)
    return zip(
        [s.id for s in game_states], [root.terminal_value] * len(game_states),
        [self.config.policy_target(state.actions) for state in game_states])

  def simulate_epoch(self, print_progress=False):
    """Simulates an epoch of games.

    Args:
      print_progress: A boolean, whether or not to print simulation progress
        updates.

    Returns:
      A list of (state, target value, target policy) tuples, the data collected
      from an epoch of game simulations.
    """
    data = []
    for game_number in xrange(self.config.games_per_epoch):
      if print_progress and game_number % 10 == 1:
        print 'Simulating game #{}'.format(game_number)
      data.extend(self.simulate_game())
    return data

  def train(self, print_progress=False, save_data_path='', load_data_path=''):
    """Trains the agent according to its config, updating its model.

    Args:
      print_progress: A boolean, whether or not to print simulation progress
        updates.
      save_data_path: A string -- if non-empty will save the training data to
        files with epoch suffixes (so that it can be reused later).
      load_data_path: A string -- if non-empty, will load previously-saved data
        from the specified path (with epoch suffixes) instead of running the
        simulations.
    """
    for e in xrange(self.config.training_epochs):
      if load_data_path:
        data = self.load_epoch('{}_{}.txt'.format(load_data_path, e))
      else:
        data = self.simulate_epoch(print_progress=print_progress)
      if save_data_path and not load_data_path:
        self.save_epoch('{}_{}.txt'.format(save_data_path, e), data)
      self.model.train(data)
      for opponent in self.match_opponents:
        gl.play_match(
            self.game,
            self,
            opponent,
            num_games=self.games_per_match,
            evaluator=self.evaluator)

  def select_move(self):
    """Selects the move to play at inference time.

    Returns:
      The integer action to be taken.
    """
    states = {}
    root = gt.State(
        states=states,
        game=self.game,
        model=self.model,
        rollout_policy=self.config.rollout_policy,
        opponent_rollout_policy=self.config.opponent_rollout_policy)
    for _ in xrange(self.config.inference_rollouts_per_move):
      root.rollout(
          self.config.inference_rollout_depth,
          current_depth=0,
          rollout_actions=[])
    return self.game.index_to_action(
        self.config.inference_policy(root.actions).action)

  def save_epoch(self, filename, data):
    """Saves an epoch of data to the provided filename.

    Args:
      filename: A string, the file path to which the data should be saved.
      data: A list of (state, target value, target_policy) tuples.
    """
    state_size = reduce(lambda x, y: x * y, self.game.state_shape())
    with open(filename, 'w') as f:
      for position, value, policy in data:
        serialized_position = ','.join(['{:0.5f}' for _ in range(state_size)
                                       ]).format(*np.reshape(position, [-1]))
        serialized_value = '{:0.5f}'.format(value)
        serialized_policy = ','.join([
            '{:0.5f}' for _ in range(self.game.action_space())
        ]).format(*policy)
        line = '{}|{}|{}\n'.format(serialized_position, serialized_value,
                                   serialized_policy)
        f.write(line)

  def load_epoch(self, filename):
    """Loads an epoch of data from the specified file path.

    Args:
      filename: Path to the data file to load.

    Returns:
      A list of (state, target value, target_policy) tuples loaded from the
      file.
    """
    def format_line(line):
      pos, val, pol = line.split('|')
      return (np.reshape(map(float, pos.split(',')), self.game.state_shape()),
              float(val), np.array(map(float, pol.split(','))))

    with open(filename, 'r') as f:
      return [format_line(line) for line in f.read().splitlines()]
