import numpy as np
from itertools import groupby


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
        """
            基于当前玩家落子前，判断当前局面是否结束，一般来说若结束且非和棋都会返回-1.0，
            因为现在轮到当前玩家落子了，但是游戏却已经结束了，结束前的最后一步一定是对手落子的，对手赢了，则返回-1
            :param board:
            :param 5:五子棋，5就等于五
            :return:
            """
        win_cand = 2 if player_color == 1 else 1
        for
        h, w = self.board_size
        for i in range(h):
            for j in range(w):
                hang = sum(self.board[i: min(i + 5, w), j])
                if hang == 5:
                    self.is_terminal = True
                    return player_color
                elif hang == -5:
                    return True, -1.0
                lie = sum(self.board[i, j: min(j + 5, h)])
                if lie == 5:
                    return True, 1.0
                elif lie == -5:
                    return True, -1.0
                # 斜线有点麻烦
                if i <= h - 5 and j <= w - 5:
                    xie = sum([self.board[i + k, j + k] for k in range(5)])
                    if xie == 5:
                        return True, 1.0
                    elif xie == -5:
                        return True, -1.0
                if i >= 5 - 1 and j <= w - 5:
                    xie = sum([self.board[i - k, j + k] for k in range(5)])
                    if xie == 5:
                        return True, 1.0
                    elif xie == -5:
                        return True, -1.0
        if np.where(self.board == 0)[0].shape[0] == 0:  # 棋盘满了，和棋
            return True, 0.0
        return False, 0.0
        return 1