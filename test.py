"""Tests for the RL libraries."""

from __future__ import division

import functools

import numpy as np

import agents as ag
import game_tree as gt
import models as mo
import policies as po
import rl_agent as rl
import tictactoe as ttt


def expect_equal(a, b, name):
  if a == b:
    print 'SATISFIED EXPECTATION: {}'.format(name)
    return True
  else:
    print 'UNSATISFIED EXPECTATION: {}'.format(name)
    print '{} is not equal to {}'.format(a, b)
    return False


def test_result(passed, name):
  if passed:
    print '{} PASSES\n'.format(name)
  else:
    print '{} FAILS\n'.format(name)


class TestModel(object):

  def value(self, *args):
    return 2.0

  def policy(self, *args):
    return dict(zip(range(9), [3.0] * 9))


class TestOpponent(object):

  def __init__(self, game):
    self.game = game

  def policy(self):
    return self.game.available_actions()[-1]


def test_game_tree():
  """Tests for the game tree."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = TestModel()
  root = gt.State({}, game, model, po.first_policy, None)
  root.rollout(9, 0, [])
  result = result and expect_equal(
      len(root.states), 8, 'visited states after first rollout')
  for i in range(9):
    result = result and expect_equal(root.actions[i].visit_count,
                                     1 if i == 0 else 0,
                                     'visit count for action {}'.format(i))
  root.rollout(9, 0, [])
  result = result and expect_equal(
      len(root.states), 8, 'visited states after second rollout')
  for i in range(9):
    result = result and expect_equal(root.actions[i].visit_count,
                                     2 if i == 0 else 0,
                                     'visit count for action {}'.format(i))
  result = result and expect_equal(root.prior, 2.0, 'prior')
  for i in range(9):
    result = result and expect_equal(root.actions[i].prior, 3.0,
                                     'prior for action {}'.format(i))
  result = result and expect_equal(root.actions[0].average_value, 1.0,
                                   'average value')
  test_result(result, 'Game Tree Test')


def test_game_tree_with_opponent():
  """Tests for the game tree with an opponent."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = TestModel()
  opponent = TestOpponent(game)
  root = gt.State({}, game, model, po.first_policy, opponent.policy)
  root.rollout(9, 0, [])
  result = result and expect_equal(
      len(root.states), 4, 'visited states after first rollout')
  result = result and expect_equal(root.actions[0].average_value, 1.0,
                                   'average value')
  test_result(result, 'Game Tree With Opponent Test')


def test_table_model():
  """Tests for the table model."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = mo.TableModel(3)
  model.insert_or_update(game.state(), (1.0, np.ones(9), 1.0))
  result = result and expect_equal(len(model.table), 1, 'table size')
  stats, rotation_amt, flipped, key = model.lookup(game.state())
  result = result and expect_equal(rotation_amt, 0, 'rotation amount')
  result = result and expect_equal(flipped, False, 'flipped')
  result = result and expect_equal(key, game.state_key(), 'table key')
  game.take_action(0)
  game.take_action(1)
  result = result and expect_equal(
      model.lookup(game.state()), None, 'non-existent key')
  original_key = game.state_key()
  model.insert_or_update(game.state(), (1.0, np.arange(9), 1.0))
  stats, rotation_amt, flipped, key = model.lookup(game.state())
  result = result and expect_equal(rotation_amt, 0, 'rotation amount')
  result = result and expect_equal(flipped, False, 'flipped')
  result = result and expect_equal(key, original_key, 'table key')
  game.undo_action()
  game.undo_action()
  game.take_action(8)
  game.take_action(5)
  stats, rotation_amt, flipped, key = model.lookup(game.state())
  result = result and expect_equal(rotation_amt, 1, 'rotation amount')
  result = result and expect_equal(flipped, True, 'flipped')
  result = result and expect_equal(key, original_key, 'table key')
  model.insert_or_update(game.state(), (3.0, np.arange(9) * 3, 1.0))
  stats, _, _, _ = model.lookup(game.state())
  value, policy, weight = stats
  result = result and expect_equal(value, 2.0, 'value')
  result = result and expect_equal(
      np.array_equal(policy, np.array([4, 4, 4, 8, 8, 8, 12, 12, 12])), True,
      'policy')
  result = result and expect_equal(weight, 2.0, 'weight')
  result = result and expect_equal(
      np.array_equal(
          model.policy(game.state()), np.array([4, 4, 4, 8, 8, 8, 12, 12, 12])),
      True, 'model policy')
  result = result and expect_equal(
      model.value(game.state()), 2.0, 'model value')
  test_result(result, 'Table Model Test')


def test_policies():
  """Tests for the policies."""
  result = True
  actions = {
      0:
          gt.Action(
              None,
              None,
              action=0,
              visit_count=9,
              average_value=0.1,
              prior=0.2,
              for_maximizer=True),
      1:
          gt.Action(
              None,
              None,
              action=1,
              visit_count=16,
              average_value=0.9,
              prior=0.1,
              for_maximizer=True),
      8:
          gt.Action(
              None,
              None,
              action=8,
              visit_count=25,
              average_value=-0.8,
              prior=0.4,
              for_maximizer=True),
      7:
          gt.Action(
              None,
              None,
              action=7,
              visit_count=50,
              average_value=0.0,
              prior=0.3,
              for_maximizer=True)
  }
  np.testing.assert_almost_equal(
      po.alpha_zero_visit_counts_to_target(actions, 9, 1.0),
      np.array([0.09, 0.16, -1.0, -1.0, -1.0, -1.0, -1.0, 0.50, 0.25]))
  np.testing.assert_almost_equal(
      po.average_values_to_target(actions, 9),
      np.array([1.1, 1.9, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 0.2]))
  result = result and expect_equal(
      po.greedy_value_policy(actions), actions[1], 'greedy value')
  result = result and expect_equal(
      po.greedy_visit_policy(actions), actions[7], 'greedy visit')
  result = result and expect_equal(
      po.greedy_prior_policy(actions), actions[8], 'greedy prior')
  result = result and expect_equal(
      po.alpha_zero_mcts_policy(actions, c_puct=1.0), actions[1],
      'alpha zero mcts')
  for a in actions.iteritems():
    a[1].for_maximizer = False
  result = result and expect_equal(
      po.greedy_value_policy(actions), actions[8], 'greedy value minimizer')
  result = result and expect_equal(
      po.alpha_zero_mcts_policy(actions, c_puct=1.0), actions[8],
      'alpha zero mcts minimizer')
  np.testing.assert_almost_equal(
      po.average_values_to_target(actions, 9),
      np.array([0.9, 0.1, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.8]))
  # Smoke test for the randomized policies.
  po.random_policy(actions)
  po.alpha_zero_play_policy(actions, tau=1.0)
  test_result(result, 'Policies Test')


def test_agents():
  """Tests for the stock agents."""
  result = True
  game = ttt.TicTacToe(3, 3)
  random = ag.RandomAgent('Random', game)
  dumb = ag.DumbAgent('Dumb', game)
  optimal = ag.MinimaxAgent('Minimax', game)
  # Smoke test for the random agent.
  game.take_action(random.select_move())
  game.undo_action()
  result = result and expect_equal(dumb.select_move(), 0, 'dumb first move')
  game.take_action(0)
  result = result and expect_equal(dumb.select_move(), 1, 'dumb second move')
  game.take_action(4)
  game.take_action(3)
  result = result and expect_equal(optimal.select_move(), 6, 'minimax move')
  result = result and expect_equal(optimal.optimal_moves(), [6],
                                   'minimax optimal moves')
  game.reset()
  # result = result and expect_equal(optimal.optimal_moves(), range(9), 'minimax all moves optimal')
  test_result(result, 'Agents Test')


def test_rl_agent():
  """Tests for the RL Agent."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = mo.TableModel(3)
  config = rl.Config(
      training_epochs=1,
      games_per_epoch=1,
      rollouts_per_move=1,
      rollout_depth=9,
      rollout_policy=functools.partial(po.alpha_zero_mcts_policy, c_puct=10.0),
      play_policy=functools.partial(po.alpha_zero_play_policy, tau=1.0),
      inference_policy=po.greedy_prior_policy,
      opponent_rollout_policy=None,
      opponent_play_policy=None,
      policy_target=functools.partial(
          po.alpha_zero_visit_counts_to_target,
          action_space=game.action_space(),
          tau=1.0),
      inference_rollouts_per_move=0,
      inference_rollout_depth=0)
  agent = rl.RLAgent('RL Agent', game, model, config, [], 0, None)
  agent.train()
  test_result(result, 'RL Agent Test')


def test_fully_connected_model():
  """Tests for the fully connected model."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = mo.FullyConnectedModel(3, [9, 9], batch_size=2)
  result = result and expect_equal(len(model.layers), 3, 'number of layers')
  s1 = game.state()
  game.take_action(0)
  s2 = game.state()
  game.take_action(4)
  s3 = game.state()
  result = result and expect_equal(model.value(s1), 0.0, 'value')
  model.train([(s1, 0, np.arange(9, dtype=np.float32)),
               (s2, 1.0, np.zeros(9, dtype=np.float32)),
               (s3, 0.5, np.ones(9, dtype=np.float32))])
  test_result(result, 'Fully Connnected Model Test')


def test_convolutional_model():
  """Tests for the convolutional model."""
  result = True
  game = ttt.TicTacToe(3, 3)
  model = mo.ConvolutionalModel(3, [10, 9], batch_size=2)
  result = result and expect_equal(len(model.layers), 2, 'number of layers')
  s1 = game.state()
  game.take_action(0)
  s2 = game.state()
  game.take_action(4)
  s3 = game.state()
  result = result and expect_equal(model.value(s1), 0.0, 'value')
  np.testing.assert_almost_equal(model.policy(s1), np.ones(9) / 9)
  model.train([(s1, 0, np.arange(9, dtype=np.float32)),
               (s2, 1.0, np.zeros(9, dtype=np.float32)),
               (s3, 0.5, np.ones(9, dtype=np.float32))])
  test_result(result, 'Convolutional Model Test')


def run_tests():
  """Runs all of the test."""
  test_game_tree()
  test_game_tree_with_opponent()
  test_table_model()
  test_policies()
  test_agents()
  test_rl_agent()
  test_fully_connected_model()
  test_convolutional_model()


run_tests()
