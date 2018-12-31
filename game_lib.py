# Runs a match between two agents.
def play_match(g, 
			   agent_a, 
			   agent_b, 
			   num_games,
			   evaluator=None, 
			   print_games=False):
	results = {agent_a.name : 0, 
			   agent_b.name : 0, 
			   "Draws" : 0}
	results_by_player = { 
			   agent_a.name+'-first':0, 
			   agent_a.name+'-second':0,
			   agent_b.name+'-first':0, 
			   agent_b.name+'-second':0}
	if evaluator is not None:
		optimality = {agent_a.name+'_optimal': 0,
					  agent_a.name+'_suboptimal': 0, 
					  agent_b.name+'_optimal': 0, 
					  agent_b.name+'_suboptimal': 0}
 	for game in range(num_games):
		g.reset()
		a_to_move_first = game % 2 == 0
		if print_games:
			x = agent_a.name if a_to_move_first else agent_b.name
			o = agent_b.name if a_to_move_first else agent_a.name
			print x, 'is playing first - ', o, 'is playing second'
		moves = []
		while g.result() is None:
			move = agent_a.select_move() if a_to_move_first == g.maximizer_to_act() else agent_b.select_move()
			if evaluator is not None:
				optimal_moves = evaluator.optimal_moves()
				key_prefix = agent_a.name if a_to_move_first == g.maximizer_to_act() else agent_b.name
				if move in optimal_moves:
					optimality[key_prefix+'_optimal'] += 1
				else:
					optimality[key_prefix+'_suboptimal'] += 1
			moves.append(move)
			g.take_action(move)
			if print_games:
				g.print_state()
				print
		if g.result() == 0:
			results["Draws"] += 1
		else:
			if a_to_move_first == g.maximizer_to_act():
				winning_agent = agent_b.name
			else:
				winning_agent = agent_a.name
			results[winning_agent] += 1
			winning_parity = 'second' if g.result() == -1.0 else 'first'
			results_by_player[winning_agent+'-'+winning_parity] +=1
		if print_games:
			print "Draw!" if g.result == 0 else winning_agent + ' as ' + winning_parity +  ' player WINS'
	overall_results = sorted(results.items())
	results_breakdown = sorted(results_by_player.items())
	print 
	print 'RESULTS - {} vs. {} '.format(agent_a.name, agent_b.name)
	for focus_name, opp_name in [[agent_b.name, agent_a.name], [agent_a.name, agent_b.name]]:
		print '{} - Wins: {} | Losses: {} | Draws: {}'.format(focus_name, results[focus_name], results[opp_name], results['Draws'])
		if evaluator is not None:
			print '{} - Optimal: {} | Sub-optimal: {} | Optimal percentage: {:.2f}%'.format(focus_name, optimality[focus_name+'_optimal'], optimality[focus_name+'_suboptimal'], 100*optimality[focus_name+'_optimal']/(optimality[focus_name+'_optimal'] + optimality[focus_name+'_suboptimal']))
	return overall_results, results_breakdown

# Play interactively against an agent in an infinite loop.
def interactive_play(g, agent):
	game = 0
	while True:
		g.reset()
		a_to_move_first = game % 2 == 0
		print 'New game!', 'Agent moves first.' if a_to_move_first else 'You move first.'
		g.print_state()
		print
		while g.result() is None:
			move = agent.select_move() if a_to_move_first == g.maximizer_to_act() else raw_input('Select a move:')
			if a_to_move_first != g.maximizer_to_act():
				try:
					move = g.parse_action(move)
				except:
					if move in ['q', 'quit']:
						return
					if move in ['r', 'resign']:
						a_to_move_first = not g.maximizer_to_act()
						break
					if move in ['u', 'undo']:
						g.undo_action()
						while a_to_move_first == g.maximizer_to_act() and g.moves:
							g.undo_action()
						g.print_state()
						continue
					print 'Please input a valid move.'
					continue
			if move in g.available_actions():
				g.take_action(move)
				g.print_state()
				print
			else:
				print g.available_actions()
				print 'Please input a legal move.'
		game += 1
		if g.result() == 0.0:
			print 'Game Drawn'
		else:
			if a_to_move_first == g.maximizer_to_act():
				print 'You win!'
			else:
				print agent.name, 'wins!'