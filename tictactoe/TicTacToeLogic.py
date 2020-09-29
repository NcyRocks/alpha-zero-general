import numpy as np
# from bkcharts.attributes import color
class Board:
    """Board class for the game of TicTacToe.
    Default board size is 3x3.
    Board data:
    1 = white(O), -1 = black(X), 0 = empty
    first dim is column , 2nd is row:
        pieces[0][0] is the top left square,
        pieces[2][0] is the bottom left square,
    Squares are stored and manipulated as (x,y) tuples.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Based on the board for the game of Othello by Eric P. Nichols.

    """

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n=3):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        pieces = [None] * self.n
        for i in range(self.n):
            pieces[i] = [0] * self.n
        self.pieces = np.array(pieces)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def is_legal_move(self, move):
        (x, y) = move
        return self[x][y] == 0

    def all_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                yield (x, y)
        return

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==0:
                    newmove = (x,y)
                    moves.add(newmove)
        return list(moves)

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==0:
                    return True
        return False
    
    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        win = self.n
        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self[d][d]==color:
                count += 1
        if count==win:
            return True
        count = 0
        for d in range(self.n):
            if self[d][self.n-d-1]==color:
                count += 1
        if count==win:
            return True
        
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color of the piece to play (1=white, -1=black).
        """

        (x,y) = move

        # Add the piece to the empty square.
        if self[x][y] == 0:
            self[x][y] = color
            return True

        return False


class InvisibleBoard(Board):
    """
    Board for a game of Invisible Tic-Tac-Toe.
    Assumes two players.
    """

    def __init__(self, n=3):
        super().__init__(n=n)
        self.visible_pieces = {player: [None] * self.n for player in (1, -1)}
        for player in (1, -1):
            for i in range(self.n):
                self.visible_pieces[player][i] = [0] * self.n
            self.visible_pieces[player] = np.array(self.visible_pieces[player])

    def execute_move(self, move, color):
        move_is_valid = super().execute_move(move, color)
        (x, y) = move
        if not move_is_valid:
            self.visible_pieces[color][x][y] = -color
        else:
            self.visible_pieces[color][x][y] = color
        return move_is_valid

    def get_legal_moves(self, color):
        """Returns all the seemingly legal moves for the given color.
        May not all be actually legal.
        (1 for white, -1 for black).
        """
        moves = set()  # stores the legal moves.

        # Get all the seemingly empty squares (color==0 or color is other player)
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x][y] != color:
                    newmove = (x,y)
                    moves.add(newmove)
        return list(moves)
