from __future__ import division

import functools

import numpy as np

import agents as ag
import game_tree as gt
import models as mo
import policies as po
import rl_agent as rl
import tictactoe as ttt


def main():
	game = ttt.TicTacToe(3, 3)
	model = mo.TableModel(3)
	evaluator = ag.MinimaxAgent('Minimax', game)
	random = ag.RandomAgent('Random', game)
	config = rl.Config(training_epochs=5, 
					   games_per_epoch=1000, 
					   rollouts_per_move=200, 
					   rollout_depth=9, 
					   rollout_policy=functools.partial(po.alpha_zero_mcts_policy, c_puct=10.0),
					   play_policy=po.random_policy,
					   inference_policy=po.greedy_prior_policy,
					   opponent_rollout_policy=None,
					   opponent_play_policy=None,
					   policy_target=functools.partial(po.alpha_zero_visit_counts_to_target, action_space=game.action_space(), tau=1.0),
					   inference_rollouts_per_move=0,
					   inference_rollout_depth=0)
	agent = rl.RLAgent('RL Agent', game, model, config, [random, evaluator], 100, evaluator)
	agent.train(print_progress=True)
	ttt.play_match(g=game, 
			   	   agent_a=agent, 
			   	   agent_b=random, 
			   	   num_games=1000, 
			       print_games=False, 
			       print_results=True, 
			       print_a_losses=True, 
			       evaluator=evaluator)

main()