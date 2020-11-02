import sys
import numpy as np

sys.path.append("..")
from Game import Game
from .Connect4Logic import *


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        Game.__init__(self)
        board = Board(height, width, win_length, np_pieces)
        self.height = board.height
        self.width = board.width
        self.win_length = board.win_length
        self.np_pieces = np_pieces

    def getInitBoard(self):
        return Board(self.height, self.width, self.win_length, self.np_pieces)

    def getBoardSize(self):
        return (self.height, self.width)

    def getActionSize(self):
        return self.width

    def getNextState(self, board, player, action):
        """Returns the board with updated move, original board is modified."""
        board.add_stone(action, player)
        return board, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        try:
            return board.get_valid_moves(player)
        except AttributeError:
            board = Board(self.height, self.width, self.win_length, board)
            return self.getValidMoves(board, player)

    def getGameEnded(self, board, player):
        try:
            winstate = board.get_win_state()
            if winstate.is_ended:
                if winstate.winner is None:
                    # draw has very little value.
                    return 1e-4
                elif winstate.winner == player:
                    return +1
                elif winstate.winner == -player:
                    return -1
                else:
                    raise ValueError("Unexpected winstate found: ", winstate)
            else:
                # 0 used to represent unfinished game.
                return 0
        # TODO: Really, really need a better workaround
        except AttributeError:
            board = Board(self.height, self.width, self.win_length, board)
            return self.getGameEnded(board, player)

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return np.copy(board.np_pieces)

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(" ".join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")

    def getModelBoard(self, canonicalBoard, player=1):
        # TODO: Rename
        return Board(self.height, self.width, self.win_length, canonicalBoard)

    def getNextPlayer(self, board):
        pieces_1 = 0
        pieces_2 = 0
        for y in range(self.width):
            for x in range(self.height):
                if board[x][y] == 1:
                    pieces_1 += 1
                if board[x][y] == -1:
                    pieces_2 += 1
        if pieces_1 > pieces_2:
            return -1
        else:
            return 1


class InvisibleConnect4Game(Connect4Game):
    
    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        super().__init__(height, width, win_length, np_pieces)
        board = InvisibleBoard(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return InvisibleBoard(self.height, self.width, self.win_length, self.np_pieces)

    def getNextState(self, board, player, action):
        if board.add_stone(action, player):
        # Weird solution that'll probably work
            next_player = self.getNextPlayer(board)
            return (board, next_player)
        else:
            return (board, player)

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return np.copy(board.visible_pieces[player])

    def getModelBoard(self, canonicalBoard, player):
        # TODO: Rename
        return InvisibleBoard(self.height, self.width, self.win_length, canonicalBoard, player)

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(" ".join(map(str, range(len(board[0])))))
        print(board)
        print(board.visible_pieces[1])
        print(board.visible_pieces[-1])
        print(" -----------------------")
