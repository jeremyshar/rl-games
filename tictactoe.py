"""Implementation of the game of tic tac toe."""

from __future__ import division

import itertools
import numpy as np

import abstract


class TicTacToe(abstract.Game):
  """Class containing an implementation of tic tac toe."""

  def __init__(self, board_size, num_in_a_row):
    # Mapping between potential game winners and outcome scores. _ is a draw.
    self.value_mapping = {'_': 0, 'X': 1.0, 'O': -1.0}
    self.empty = '_'
    self.winner = self.empty
    self.players = ['X', 'O']
    self.first_player_to_move = True
    self.board_size = board_size
    self.num_in_a_row = num_in_a_row
    # Initialize the board.
    self.board = [
        [[self.empty] for i in range(board_size)] for j in range(board_size)
    ]
    # Columns and diagonals are just for semi-efficient indexing of the game.
    self.columns = [
        [self.board[j][i] for j in range(board_size)] for i in range(board_size)
    ]
    self.diagonals = [[] for i in range((board_size * 4) - 2)]
    for i in range(board_size):
      for j in range(board_size):
        self.diagonals[i + j].append(self.board[i][j])
        self.diagonals[(board_size - i - 1) + j + (2 * board_size) - 1].append(
            self.board[i][j])
    self.moves = []

  def result(self):
    # Iterate through the rows, columns and diagonals and check for num_in_a_row
    # in a row.
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
    return self.value_mapping[self.empty] if self.available_actions() else None

  def print_state(self):
    for row in self.board:
      print '|' + '|'.join([elt[0] for elt in row]) + '|'

  def available_actions(self):
    return [
        i for i in range(self.board_size**2)
        if self.board[i // self.board_size][i %
                                            self.board_size][0] == self.empty
    ]

  def take_action(self, action):
    self.board[action // self.board_size][
        action % self.board_size][0] = 'X' if self.first_player_to_move else 'O'
    self.moves.append(action)
    self.first_player_to_move = not self.first_player_to_move

  def undo_action(self):
    move = self.moves.pop()
    self.winner = self.board[move // self.board_size][
        move % self.board_size][0] = self.empty
    self.first_player_to_move = not self.first_player_to_move

  def reset(self):
    self.winner = self.empty
    self.first_player_to_move = True
    self.moves = []
    for i in range(self.board_size):
      for j in range(self.board_size):
        self.board[i][j][0] = self.empty

  def state_shape(self):
    return (self.board_size, self.board_size, 3)

  def state(self):
    stack = np.zeros(shape=self.state_shape())
    if not self.first_player_to_move:
      stack[:, :, 2] = 1.0
    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.board[i][j][0] == 'X':
          stack[i][j][0] = 1.0
        elif self.board[i][j][0] == 'O':
          stack[i][j][1] = 1.0
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
    return self.board[index // self.board_size][index % self.board_size][0]

  def action_space(self):
    return self.board_size * self.board_size

  def action_index(self, action):
    return action

  def index_to_action(self, index):
    return index

  def parse_action(self, action_string):
    return int(action_string)
