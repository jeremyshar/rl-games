from __future__ import division

import functools

import numpy as np
import tensorflow as tf

import agents as ag
import game_lib as gl
import game_tree as gt
import policies as po
import rl_agent as rl
import tictactoe as ttt
import checkers as ck

import random as rn

import models as mo


def main():
	game = ttt.TicTacToe(8, 4)
	random = ag.RandomAgent('Random', game)
	config = rl.Config(training_epochs=2, 
					   games_per_epoch=10, 
					   rollouts_per_move=20, 
					   rollout_depth=4, 
					   rollout_policy=functools.partial(po.alpha_zero_mcts_policy, c_puct=10.0),
					   play_policy=functools.partial(po.alpha_zero_play_policy, tau=1.5),
					   inference_policy=po.greedy_prior_policy,
					   opponent_rollout_policy=None,
					   opponent_play_policy=None,
					   policy_target=functools.partial(po.alpha_zero_visit_counts_to_target, action_space=game.action_space(), tau=1.0),
					   inference_rollouts_per_move=40,
					   inference_rollout_depth=4)
	model = mo.KerasModel(game, [128], [64, 32], [16, 4], data_passes=100)
	agent = rl.RLAgent('RL Agent', game, model, config, [random], 100)
	agent.train(print_progress=True)
	gl.play_match(g=game,
				   agent_a=agent,
				   agent_b=random,
				   num_games=4)
	gl.interactive_play(game, agent)

main()