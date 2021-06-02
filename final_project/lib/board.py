import logging
import numpy as np
from itertools import groupby


class Board:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))
        self.is_terminal = False

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.is_terminal = False

    def get_legal_coords(self):
        legal_coords = []
        _board = self.board.copy().reshape(-1)
        for pos, val in enumerate(_board):
            if val == 0:
                legal_coords.append(pos)
        return legal_coords

    def play(self, coord, player_color):
        self.board[coord[0], coord[1]] = player_color
        # logging.info("after {}".format(self.board))

    def get_color(self, player_color):
        _board = self.board.copy()
        for i in range(_board.shape[0]):
            for j in range(_board.shape[1]):
                if _board[i, j] != player_color:
                    _board[i, j] = 0
        return _board

    def check_terminal(self, player_color):
        # TODO 调回五子棋
        def five(array):
            winner = None
            dic = {}
            for k, g in groupby(array):
                s = sum(1 for _ in g)
                if k in dic and s <= dic[k]:
                    continue
                else:
                    dic[k] = s
            if 1 in dic and dic[1] >= 4:
                winner = 1
            elif 2 in dic and dic[2] >= 4:
                winner = 2
            if winner is not None and winner == player_color:
                logging.warning("winner:{}, player_color:{}".format(winner,player_color))
                logging.warning("Something went wrong! Maybe wrong winner!")
                logging.warning(dic)
            return winner

        for i in range(self.board_size):
            winner = five(self.board[i, :])
            if winner is not None:
                self.is_terminal = True
                return winner

            winner = five(self.board[:, i])
            if winner is not None:
                self.is_terminal = True
                return winner
        # TODO
        # for i in range(-self.board_size + 5, self.board_size - 4):
        for i in range(-self.board_size + 4, self.board_size - 3):
            winner = five(self.board.diagonal(offset=i))
            if winner is not None:
                self.is_terminal = True
                return winner

            winner = five(np.diag(np.fliplr(self.board), i))
            if winner is not None:
                self.is_terminal = True
                return winner

        if not np.sum(np.array(self.board > 0)):
            self.is_terminal = True
            return 2
        return None
