import numpy as np
import random
from torch.utils.data import Dataset
from utils.config import *
from utils import utils


class SelfPlayDataset(Dataset):
    """
    Self-play dataset containing state, probabilities
    and the winner of the game.
    """

    def __init__(self):
        """ Instanciate a dataset """

        self.states = np.zeros((MOVES, (HISTORY + 1) * 2 + 1, GOBANG_SIZE, GOBANG_SIZE))
        self.plays = np.zeros((MOVES, GOBANG_SIZE ** 2))
        self.winners = np.zeros(MOVES)
        self.current_len = 0

    def __len__(self):
        return self.current_len

    def __getitem__(self, idx):
        return utils.formate_state(self.states[idx], self.plays[idx], self.winners[idx])

    def update(self, game):
        """ Rotate the circular buffer to add new games at end """

        moves = len(game[0])
        final_moves = moves
        sample = random.sample([i for i in range(moves)], final_moves)
        self.current_len = min(self.current_len + final_moves, MOVES)
        self.states = np.roll(self.states, final_moves, axis=0)
        self.states[:final_moves] = np.vstack(tuple(game[0][i][0] for i in sample))
        self.plays = np.roll(self.plays, final_moves, axis=0)
        self.plays[:final_moves] = np.vstack(tuple(game[0][i][1] for i in sample))

        # Replace the player color with the end game result
        players = np.array([game[0][i][2] for i in sample])
        players[np.where(players - 1 != game[1])] = -1
        players[np.where(players != -1)] = 1

        self.winners = np.roll(self.winners, final_moves, axis=0)
        self.winners[:final_moves] = players

        return final_moves