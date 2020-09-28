from __future__ import print_function
import sys

sys.path.append("..")
from Game import Game
from .TicTacToeLogic import *
import numpy as np


class TicTacToeGame(Game):
    """Game class implementation for the game of TicTacToe.
    Based on the OthelloGame then getGameEnded() was adapted to new rules.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Based on the OthelloGame by Surag Nair.
    """

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
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board, player)
        # action need not be a valid move - invalid moves will prompt another turn
        if action == self.n * self.n:
            return (board, -player)
        move = (int(action / self.n), action % self.n)
        # TODO: This 'if' statement is not necessary now
        if board.execute_move(move, player):
            return (board, -player)
        else:
            return (board, player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector

        # Can receive an array or a board as input, due to compatibility reasons
        # TODO: Find a better solution
        valids = [0] * self.getActionSize()
        try:
            legalMoves = board.get_legal_moves(player)
        except AttributeError:
            moves = set()  # stores the legal moves.

            # Get all the empty squares (color==0)
            for y in range(self.n):
                for x in range(self.n):
                    if board[x][y] == 0:
                        newmove = (x, y)
                        moves.add(newmove)
            legalMoves = list(moves)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        # Yet another crappy workaround - bear with
        try:
            if board.is_win(player):
                return 1
            if board.is_win(-player):
                return -1
            if board.has_legal_moves():
                return 0
            # draw has a very little value
            return 1e-4
        except AttributeError:
            board = Board(self.n, board)
            return self.getGameEnded(board, player)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board.pieces

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n ** 2 + 1  # 1 for pass
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
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board.pieces[y][x]  # get the piece to print
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")

    def getModelBoard(self, canonicalBoard):
        # TODO: Rename
        return Board(self.n, canonicalBoard)


class InvisibleTicTacToeGame(TicTacToeGame):
    """Implementation of Invisible TicTacToe.
    Neither player can see what move their opponent makes.
    If a player tries to make an illegal move, they are told it is illegal and their board is updated.
    """

    def getInitBoard(self):
        return InvisibleBoard(self.n)

    def getCanonicalForm(self, board, player):
        # return visible state if player==1, else return -state if player==-1
        return board.visible_pieces[player] * player

    def getPossibleOutcomes(self, board, player, action):
        # Board should only be a np array, not an actual board, for I-didn't-write-this-I'm-just-tying-things-together-that-shouldn't-go-together reasons
        empties = 0
        other_pieces = 0
        my_pieces = 0
        for y in range(self.n):
            for x in range(self.n):
                if board[x][y] == 0:
                    empties += 1
                if board[x][y] == -player:
                    other_pieces += 1
                if board[x][y] == player:
                    my_pieces += 1
        hidden_pieces = my_pieces - other_pieces
        if player == -1:
            hidden_pieces += 1
        move = (int(action / self.n), action % self.n)
        (x, y) = move
        if_valid = np.copy(board)
        if_valid[x][y] = player
        if_invalid = np.copy(board)
        if_invalid[x][y] = -player
        valid_odds = (empties - hidden_pieces) / empties
        invalid_odds = hidden_pieces / empties
        # if self.getGameEnded(if_invalid):
        #     valid_odds = 1
        #     invalid_odds = 0
        return [(if_valid, valid_odds), (if_invalid, invalid_odds)]

    def getModelBoard(self, canonicalBoard):
        # TODO: Rename
        return InvisibleBoard(self.n, canonicalBoard)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board, player)
        # action need not be a valid move - invalid moves will prompt another turn

        # TODO: Instead of checking valid moves, check whose next turn it is based on numbers of steps
        if action == self.n * self.n:
            return (board, -player)
        move = (int(action / self.n), action % self.n)
        if board.execute_move(move, player):
        # Weird solution that'll probably work
            pieces_1 = 0
            pieces_2 = 0
            for y in range(self.n):
                for x in range(self.n):
                    if board[x][y] == 1:
                        pieces_1 += 1
                    if board[x][y] == -1:
                        pieces_2 += 1
            if pieces_1 > pieces_2:
                return (board, -1)
            else:
                return (board, 1)
        else:
            return (board, player)

    @staticmethod
    def display(board):
        n = board.pieces.shape[0]
        boards = [board.pieces, board.visible_pieces[1], board.visible_pieces[-1]]

        extra_spaces = "  " * (n - 3)
        print("|Board    ", end="")
        print(extra_spaces, end="")
        print("| Player 1 ", end="")
        print(extra_spaces, end="")
        print("| Player -1", end="")
        print(extra_spaces, end="")
        print("|")

        for _ in range(3):
            print("   ", end="")
            for y in range(n):
                print(y, "", end="")
            print(" ", end="")
        print("")

        for _ in range(3):
            print("  ", end="")
            print("--" * (n + 1), end="")
            print(" ", end="")
        print("")

        for y in range(n):
            for i in range(3):
                print(y, "|", end="")  # print the row #
                for x in range(n):
                    piece = boards[i][y][x]  # get the piece to print
                    if piece == -1:
                        print("X ", end="")
                    elif piece == 1:
                        print("O ", end="")
                    else:
                        if x == n:
                            print("-", end="")
                        else:
                            print("- ", end="")
                if i == 2:
                    print("|")
                else:
                    print("| ", end="")

        for _ in range(3):
            print("  ", end="")
            print("--" * (n + 1), end="")
            print(" ", end="")
        print("")
