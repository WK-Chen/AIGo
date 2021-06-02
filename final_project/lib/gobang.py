import numpy as np
import sys
import six
from utils.config import HISTORY
from lib.board import Board
import logging


def _action_to_coord(board, a):
    return a // board.board_size, a % board.board_size


def _format_state(history, player_color, board_size):
    """
    Format the encoded board into the state that is the input
    of the feature model, defined in the AlphaGo Zero paper
    BLACK = 1
    WHITE = 2
    """

    state_history = np.concatenate((history[0], history[1]), axis=0)
    to_play = np.full((1, board_size, board_size), player_color - 1)
    final_state = np.concatenate((state_history, to_play), axis=0)
    return final_state


def stone_other(player_color):
    return player_color + 1 if player_color == 1 else player_color - 1


class GobangEnv():

    def __init__(self, player_color, board_size):
        self.board_size = board_size
        self.history = [np.zeros((HISTORY + 1, board_size, board_size)),
                        np.zeros((HISTORY + 1, board_size, board_size))]

        colormap = {
            'black': 1,
            'white': 2,
        }
        self.player_color = colormap[player_color]

        self.board = Board(self.board_size)
        self.state = _format_state(self.history, self.player_color, self.board_size)
        self.done = True

    def get_legal_moves(self):
        """ Get all the legal moves and transform their coords into 1d """

        legal_moves = self.board.get_legal_coords()
        return legal_moves

    def _act(self, action, history):
        """ Executes an action for the current player """
        self.board.play(_action_to_coord(self.board, action), self.player_color)
        color = self.player_color - 1
        history[color] = np.roll(history[color], 1, axis=0)
        history[color][0] = self.board.get_color(self.player_color)
        self.player_color = stone_other(self.player_color)

    def reset(self):
        """ Reset the board """
        self.board.reset()
        opponent_resigned = False
        self.done = self.board.is_terminal or opponent_resigned
        return _format_state(self.history, self.player_color, self.board_size)

    def get_reward(self, winner):
        return 0 if winner == 1 else 1

    def step(self, action):
        """ Perfoms an action and choose the winner if the 2 player have passed """
        # logging.info("player:{} ACTION: {}".format(self.player_color, action))

        if not self.done:
            self._act(action, self.history)

        winner = self.board.check_terminal(self.player_color)
        # Reward: if nonterminal, then the reward is -1
        if not self.board.is_terminal:
            return _format_state(self.history, self.player_color, self.board_size), -1, False

        assert self.board.is_terminal
        self.done = True
        reward = self.get_reward(winner)
        return _format_state(self.history, self.player_color, self.board_size), reward, self.done