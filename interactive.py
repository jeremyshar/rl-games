import agents as ag
import checkers as ck
import tictactoe as ttt
import game_lib as gl

def main():
	game = ttt.TicTacToe(8, 4)
	random = ag.RandomAgent('Random', game)
	print game.action_space()
	gl.interactive_play(game, random)

main()