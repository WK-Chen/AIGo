import numpy as np
from collections import Counter


class Board():
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))
        self.is_terminal = False

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.is_terminal = False

    def get_legal_coords(self, player_color):
        legal_coords = []
        _board = self.board.reshape(-1)
        for pos, val in enumerate(_board):
            if val == 0:
                legal_coords.append(val)
        return legal_coords

    def play(self, coord, player_color):
        self.board[coord[0], coord[1]] = player_color

    def get_color(self, player_color):
        _board = self.board
        for i in range(_board.shape[0]):
            for j in range(_board.shape[1]):
                if _board[i, j] != player_color:
                    _board[i, j] = 0
        return _board

    def check_terminal(self, player_color):
        # TODO code to judge whether player wins
        self.is_terminal = True
        print("IS_terminal:{}".format(self.is_terminal))
        return 1