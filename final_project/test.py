### tmp
import time
from utils.utils import get_player
from lib.game import Game

def self_play(current_time, loaded_version):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves probabilities
    """

    game_id = 0
    current_version = 1
    player = False

    while True:

        new_player, checkpoint = get_player(current_time, current_version)
        if new_player:
            current_version = checkpoint['version'] + 1
            print("current_version:{}".format(current_version))

        ## Waiting for the first player to be saved
        print("[PLAY] Current improvement level: %d" % current_version)
        if current_version == 1 and not player and not new_player:
            print("[PLAY] Waiting for first player")
            time.sleep(5)
            continue

        if new_player:
            player = new_player
            print("[PLAY] New player !")

        game = Game(player, 1, opponent=None)
        game.__call__()
        print("FINISH")
        assert False

if __name__ == '__main__':
    self_play("1620546596", None)
