from lib.rule import Rule
from utils.utils import load_player
from lib.game import Game
from utils.config import *
from time import sleep
import pickle

def main(round, best_player=False):
    player1, _ = load_player(49)
    player2, _ = load_player(None, best_player=True)
    x = Game(player1, 0, opponent=player2).__call__()
    print(pickle.loads(x))


if __name__ == '__main__':
    main(round=4, best_player=False)
