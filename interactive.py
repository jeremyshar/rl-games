"""Program to play a match against an agent."""

import agents as ag
import checkers as ck
import game_lib as gl
import tictactoe as ttt


def main():
  game = ck.Checkers(8, 3)
  random = ag.RandomAgent('Random', game)
  print game.action_space()
  gl.interactive_play(game, random)


main()
