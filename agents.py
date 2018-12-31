"""A library of stock agents for playing games."""

from __future__ import division

import abstract

import numpy as np


class RandomAgent(abstract.Agent):
  """An agent that makes completely random moves."""

  def __init__(self, name, game):
    self.name = name
    self.game = game

  def select_move(self):
    actions = self.game.available_actions()
    chosen_index = np.random.choice(len(actions))
    return actions[chosen_index]


class DumbAgent(abstract.Agent):
  """An agent that picks the first legal move."""

  def __init__(self, name, game):
    self.name = name
    self.game = game

  def select_move(self):
    return self.game.available_actions()[0]


class MinimaxAgent(abstract.Agent):
  """An agent that plays optimally using minimax.

  This expands the whole game tree, so it's not recommended for complicated
  games.
  """

  def __init__(self, name, game):
    self.name = name
    self.game = game
    self.cache = {}

  def select_move(self):
    _, moves = self.minimax()
    chosen_index = np.random.choice(len(moves))
    return moves[chosen_index]

  def minimax(self):
    """Returns a the value of the current state and the optimal actions."""
    # If this result is cached, immediately return it.
    state = self.game.state_key()
    if state in self.cache:
      return self.cache[state]
    # If the game is over, return the value of the result.
    if self.game.result() is not None:
      return self.game.result(), None
    # Initialize the best value to +/-infinity.
    best_value = -float('inf') if self.game.maximizer_to_act() else float('inf')
    best_moves = []
    for move in self.game.available_actions():
      # Get the value of the subtree.
      self.game.take_action(move)
      value, _ = self.minimax()
      self.game.undo_action()
      old_best_value = best_value
      # Keep track of the best move.
      best_func = max if self.game.maximizer_to_act() else min
      best_value = best_func(best_value, value)
      if old_best_value != best_value:
        best_moves = [move]
      elif value == best_value:
        best_moves.append(move)
    # Only cache part of the tree to save memory.
    if len(self.game.moves) < 5:
      self.cache[state] = (best_value, best_moves)
    return best_value, best_moves

  def optimal_moves(self):
    """Returns a list of the optimal actions in the current state."""
    _, moves = self.minimax()
    return moves
