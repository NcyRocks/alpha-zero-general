from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import *
import numpy as np

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
class TicTacToeGame(Game):
    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # return initial board (NOT numpy board, but Board itself)
        return Board(self.n)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action need not be a valid move - invalid moves will prompt another turn
        if action == self.n*self.n:
            return (board, -player)
        move = (int(action/self.n), action%self.n)
        if board.execute_move(move, player):
            return (board, -player)
        else:
            return (board, player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector

        #Can receive an array or a board as input, due to compatibility reasons
        # TODO: Find a better solution
        valids = [0]*self.getActionSize()
        try:
            legalMoves = board.get_legal_moves(player)
        except AttributeError:
            moves = set()  # stores the legal moves.

            # Get all the empty squares (color==0)
            for y in range(self.n):
                for x in range(self.n):
                    if board[x][y]==0:
                        newmove = (x,y)
                        moves.add(newmove)
            legalMoves = list(moves)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        if board.is_win(player):
            return 1
        if board.is_win(-player):
            return -1
        if board.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board.pieces

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.pieces.shape[0]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board.pieces[y][x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")


class InvisibleTicTacToeGame(TicTacToeGame):

    def getInitBoard(self):
        return InvisibleBoard(self.n)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board.visible_pieces[player]
