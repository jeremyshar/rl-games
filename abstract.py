"""Abstract classes for RL games."""

import abc


class Game(object):
  """Class representing a game to be played by the agent."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def result(self):
    """Returns the numerical result of the game, or None if the game isn't over.

    Gets the result of the game e.g. for chess, -1.0 if Black wins, 1.0 if White
    wins, 0.0 if the game is drawn, and None if the game isn't over.

    Returns:
      None if the game isn't over, or the float value of the game if it is over.
    """
    pass

  @abc.abstractmethod
  def available_actions(self):
    """Gets the actions available to the agent in the current state.

    Actions are represented as integers, and should be returned in sorted order.
    For example, in tic-tac-toe, each square could be represented as a number 0
    through 8, and the available actions would be a list of the empty squares'
    numbers in ascending order.

    Returns:
      A list of the available actions.
    """
    pass

  @abc.abstractmethod
  def take_action(self, action):
    """Mutates the current state by taking the input action.

    The action is assumed to be available/legal.

    Args:
      action: The action to take, an integer.
    """
    pass

  @abc.abstractmethod
  def undo_action(self):
    """Undoes the last action taken."""
    pass

  @abc.abstractmethod
  def reset(self):
    """Resets the game to its start state."""
    pass

  @abc.abstractmethod
  def state_shape(self):
    """Gets the shape of the game states.

    This is the shape of the tensor that represents a single game state. For
    example, one way to represent tic tac toe is with a 3x3x3 boolean tensor
    where the first channel has the locations of Xs, the second channel has the
    locations of the Os, and the third channel is all the same value
    representing which player is to move. In this case, state_shape would return
    the tuple (3, 3, 3).

    Returns:
      The shape of the state space, an iterable of integers.
    """
    pass

  @abc.abstractmethod
  def state(self):
    """Gets a tensor representation of the current game state.

    Every state for the game should be a tensor of the same shape.

    Returns:
      The state as a numpy array.
    """
    pass

  @abc.abstractmethod
  def state_key(self):
    """Gets a hashable representation of the current game state.

    Note that this key should uniquely identify the state, since states with the
    same key will be treated as the same.

    Returns:
      A hashable representation of the current state.
    """
    pass

  @abc.abstractmethod
  def print_state(self):
    """Prints the current state of the game."""
    pass

  @abc.abstractmethod
  def maximizer_to_act(self):
    """Returns True if it's the maximizer's turn to act in the current state."""
    pass

  @abc.abstractmethod
  def action_space(self):
    """Returns an int, the size of the game's action space."""
    pass

  @abc.abstractmethod
  def action_index(self, action):
    """Gets the index in action space of the given action.

    So that we can represent an agent's policy as a vector, we require that
    actions are laid out in a single dimension. For some games, it may be more
    convenient to represent actions as something more complex than a single
    integer. This method converts from a complex action to the integer action --
    in some cases this might be the identity function.

    Args:
      action: The game's internal representation of an action.

    Returns:
      An int, the index of the action in action space.
    """
    pass

  @abc.abstractmethod
  def index_to_action(self, index):
    """Returns the action corresponding to the given index in action space.

    The inverse of action_to_index, constructs a complex action from an integer
    action integer. Again, this may be the identity function.

    Args:
      index: An integer action.

    Returns:
      An action, whatever the game's internal representation is.
    """
    pass

  @abc.abstractmethod
  def parse_action(self, action_string):
    """Parses an action given as user input during interactive play.

    To allow for interactive play, it may be convenient to allow string action
    input that's different from the internal action representation. This method
    parses a user-input string as an action.

    Args:
      action_string: A user-input string representing an action.

    Returns:
      An action, whatever the game's internal representation is.
    """
    pass


class Agent(object):
  """Abstract class representing an agent that plays a game."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def select_move(self):
    """Returns the integer action to take in the current state."""
    pass


class Model(object):
  """Abstract class representing value/policy functions for an agent."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def value(self, state):
    """Computes the value/predicted outcome of the current state.

    Args:
      state: A game state.

    Returns:
      A float, the predicted result of the game from the input state.
    """
    pass

  @abc.abstractmethod
  def policy(self, state):
    """Computes a policy for the given game state.

    Args:
      state: A game state.

    Returns:
      A 1-D numpy array of values the size of the action space. The values
      should be probabilities i.e. they should sum to 1. Note that it's not
      strictly necessary for the value for an action to be 0 if the action is
      illegal.
    """
    pass

  @abc.abstractmethod
  def train(self, data):
    """Updates the model with the provided data.

    Args:
      data: A list of (state, target_value, target_policy) tuples.
    """
    pass
