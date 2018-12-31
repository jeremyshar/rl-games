"""Library containing policy functions."""

from __future__ import division

import numpy as np


def random_policy(actions):
  """Randomly selects an action.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    A randomly selected Action.
  """
  return actions[np.random.choice(actions.keys())]


def first_policy(actions):
  """Selects the first action.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    The first Action.
  """
  return actions[sorted(actions.keys())[0]]


def last_policy(actions):
  """Selects the last action.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    The last Action.
  """
  return actions[sorted(actions.keys())[-1]]


def greedy_value_policy(actions):
  """Greedily selects the action with the most favorable value.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    The Action with the highest average value.
  """
  raw_actions = actions.values()
  optimal = max if raw_actions[0].for_maximizer else min
  return optimal(raw_actions, key=lambda a: a.average_value)


def greedy_visit_policy(actions):
  """Greedily selects the action with the highest visit count.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    The Action with the highest visit count.
  """
  return max(actions.iteritems(), key=lambda a: a[1].visit_count)[1]


def greedy_prior_policy(actions):
  """Greedily selects the action with the highest prior value.

  Args:
    actions: A dict[int, Action] from action keys to Actions.

  Returns:
    The Action with the highest prior value.
  """
  return max(actions.iteritems(), key=lambda a: a[1].prior)[1]


def alpha_zero_mcts_policy(actions, c_puct):
  """Selects an action according to the AlphaZero MCTS policy.

  Since this policy has an extra parameter, it should be bound with a lambda or
  partial.

  Args:
    actions: A dict[int, Action] from action indices to Actions.
    c_puct: A float parameter controlling exploration vs. exploitation. The
      higher the value, the more the policy favors exploration.

  Returns:
    The Action with the highest score according to the AlphaZero
    MCTS policy.
  """
  raw_actions = actions.values()
  sqrt_total_visits = np.sqrt(sum(a.visit_count for a in raw_actions))
  multiplier = 1.0 if raw_actions[0].for_maximizer else -1.0
  return max(
      raw_actions,
      key=lambda a: multiplier * a.average_value + (
          c_puct * a.prior * sqrt_total_visits / (1 + a.visit_count)))


def alpha_zero_play_policy(actions, tau):
  """Selects an action according to the AlphaZero play policy.

  Since this policy has an extra parameter, it should be bound with a lambda or
  partial.

  Args:
    actions: A dict[int, Action] from action indices to Actions.
    tau: A float parameter controlling exploration vs. exploitation. The higher
      the value, the more the policy favors exploration.

  Returns:
    An Action sampled randomly from the distribution described by the
    AlphaZero play policy
  """
  numerators = np.power([a.visit_count for a in actions.values()], 1 / tau)
  probs = numerators / np.sum(numerators)
  return actions[np.random.choice(actions.keys(), p=probs)]


def alpha_zero_visit_counts_to_target(actions, action_space, tau):
  """Produces a target policy vector based on the action visit counts.

  Since this policy has extra parameters, it should be bound with a lambda or
  partial.

  Args:
    actions: A dict[int, Action] from action indices to Actions.
    action_space: An int, the number of actions in the game.
    tau: A float parameter controlling exploration vs. exploitation. The higher
      the value, the more the policy favors exploration.

  Returns:
    An array, the target policy vector, with -1 where the action isn't legal.
  """
  raw_actions = actions.values()
  numerators = np.power([a.visit_count for a in raw_actions], 1 / tau)
  probs = numerators / np.sum(numerators)
  result = -1.0 * np.ones(action_space)
  for i in xrange(len(raw_actions)):
    result[raw_actions[i].action] = probs[i]
  return result


def average_values_to_target(actions, action_space):
  """Produces a target policy vector based on the action average values.

  Since this policy has an extra parameter, it should be bound with a lambda or
  partial.

  Args:
    actions: A dict[int, Action] from action indices to Actions.
    action_space: An int, the number of actions in the game.

  Returns:
    An array, the target policy vector, with -1 where the action isn't legal.
  """
  raw_actions = actions.values()
  multiplier = 1.0 if raw_actions[0].for_maximizer else -1.0
  result = -1.0 * np.ones(action_space)
  for action in raw_actions:
    result[action.action] = 1 + (multiplier * action.average_value)
  return result
