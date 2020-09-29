#import Arena
#from MCTS import MCTS
#from othello.OthelloGame import OthelloGame
#from othello.OthelloPlayers import *
#from othello.pytorch.NNet import NNetWrapper as NNet


#import Arena_mcts as Arena # mcts use
import Arena # use this if no MCTS otherwise comment out above and below line.
from Imp_MCTS import Imp_MCTS as MCTS

# No MCTS
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as NNet

# MCTS
#from tictactoe_mcts.TicTacToeGame import InvisibleTicTacToeGame as TicTacToeGame
#from tictactoe_mcts.TicTacToePlayers import *
#from tictactoe_mcts.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

#if mini_othello:
#    g = OthelloGame(6)
#else:
#    g = OthelloGame(8)

## tic-tac-toe
g = TicTacToeGame()


# all players
rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
#hp = HumanOthelloPlayer(g).play



# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else: # 30it is actually represents 15it for imperfect no MCTS
    n1.load_checkpoint('./tto_20ep_15it_perfect/','best.h5')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, 1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)) # for MCTS
n1p = lambda x: np.argmax(n1.predict(x)[0] * g.getValidMoves(x, 1)) # noMCTS
#n1p = MCTS(g, n1, 1, args1)

if human_vs_cpu:
    player2 = hp
else:
    #n2 = NNet(g)
    #n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    #args2 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    #mcts2 = MCTS(g, n2, args2)
    #n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    #player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
    #player2 = n2
    player2 = rp # set to random player

#arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)
arena = Arena.Arena(n1p, player2, g, display=TicTacToeGame.display)

p1win = 0
p2win = 0
draw = 0
iterations = 5
for i in range(iterations):
    #print(arena.playGames(100, verbose=True))
    res = arena.playGames(100, verbose=True)
    p1win += res[0]
    p2win += res[1]
    draw += res[2]

#print("P1 Avg win: %.4f. P2 Avg win: %.4f. Draw: $.4f" % (p1win/iterations, p2win/iterations, draw/iterations))
print("P1 Avg win:", p1win/iterations)
print("P2 Avg win:", p2win/iterations)
print("Draw:", draw/iterations)
