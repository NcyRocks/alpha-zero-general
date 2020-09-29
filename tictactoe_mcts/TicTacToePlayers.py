import numpy as np
from random import choices

"""
Random and Human-ineracting players for the game of TicTacToe.
Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.
Based on the OthelloPlayers by Surag Nair.
"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game
        self.number = -1

    def play(self, board):
        def set_num(self, number):
            self.number = number
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
        


class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class NaiveNNetPlayer():
    def __init__(self, game, nnet, number):
        self.game = game
        self.nnet = nnet
        self.number = number

    def moves(self, board):
        return self.nnet.predict(board)[0] * self.game.getValidMoves(board, self.number)

    def play(self, board):
        return self.play_random(board)

    def play_max(self, board):
        return np.argmax(self.moves(board))

    def play_random(self, board):
        moves = self.moves(board)
        total = sum(moves)
        normalised = [move/total for move in moves]
        choice = choices(range(self.game.getActionSize()), k=1, weights=normalised)
        return choice[0]

    def set_num(self, number):
        self.number = number

    def __call__(self, board):
        return self.play(board)