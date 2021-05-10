from copy import deepcopy
import numpy as np
import sys
import six
from model.config import HISTORY
from lib.board import Board


def _pass_action(board_size):
    return board_size ** 2


def _resign_action(board_size):
    return board_size ** 2 + 1


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
    print("Refresh state!")
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

        legal_moves = self.board.get_legal_coords(self.player_color)
        return legal_moves

    def _act(self, action, history):
        """ Executes an action for the current player """
        self.board.play(_action_to_coord(self.board, action), self.player_color)
        color = self.player_color - 1
        history[color] = np.roll(history[color], 1, axis=0)
        history[color][0] = self.board.get_color(self.player_color)
        self.player_color = stone_other(self.player_color)

    def test_move(self, action):
        """
        Test if a specific valid action should be played,
        depending on the current score. This is used to stop
        the agent from passing if it makes him loose
        """

        board_clone = self.board.clone()
        current_score = board_clone.fast_score + self.komi

        board_clone = board_clone.play(_action_to_coord(board_clone, action), self.player_color)
        new_score = board_clone.fast_score + self.komi

        if self.player_color - 1 == 0 and new_score >= current_score \
                or self.player_color - 1 == 1 and new_score <= current_score:
            return False
        return True

    def reset(self):
        """ Reset the board """
        self.board.reset()
        opponent_resigned = False
        self.done = self.board.is_terminal or opponent_resigned
        return _format_state(self.history, self.player_color, self.board_size)

    def render(self):
        """ Print the board for human reading """

        outfile = sys.stdout
        outfile.write('To play: {}\n{}\n'.format(six.u(
            pachi_py.color_to_str(self.player_color)),
            self.board.__repr__().decode()))
        return outfile

    def get_reward(self, winner):
        return 0 if winner == 1 else 1

    def step(self, action):
        """ Perfoms an action and choose the winner if the 2 player have passed """

        if not self.done:
            self._act(action, self.history)

        winner = self.board.check_terminal(self.player_color)

        # Reward: if nonterminal, then the reward is -1
        if not self.board.is_terminal:
            return _format_state(self.history, self.player_color, self.board_size), -1, False

        assert self.board.is_terminal
        self.done = True
        reward = self.get_reward(winner)
        print("Here I am")
        return _format_state(self.history, self.player_color, self.board_size), reward, True