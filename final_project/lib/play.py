import logging
import pickle
import os
import time
import timeit
from tqdm import trange
from copy import deepcopy
from utils.config import *
from utils.utils import get_player, load_player
from .process import create_matches
from lib.game import Game


def self_play(round, target_round):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    logging.info("Start self_play()")
    while round < target_round:
        # Load the player when restarting training
        logging.info("Loading Player")
        while not os.path.isdir('./saved_models/{}'.format(round-1)):
            logging.info("self_play process sleeping")
            time.sleep(60)
        time.sleep(5)
        if round != 0:
            player, _ = load_player(round-1)
        else:
            player = get_player()

        logging.info("Loaded a player from round {}".format(round-1))

        if not player:
            logging.error("Player didn't load correctly!")
            return
        if not os.path.isdir("data/{}".format(round)):
            os.mkdir("data/{}".format(round))
        # Create the self-play match
        logging.info("Creating matches ...")
        queue, results = create_matches(player, opponent=None, match_number=SIMULATION_PER_ROUND, cores=4)
        try:
            queue.join()
            for game_id in range(SIMULATION_PER_ROUND):
                result = results.get()
                if result:
                    with open("data/{}/id_{}".format(round, game_id), 'wb') as f:
                        f.write(result)
        finally:
            queue.close()
            results.close()
        round += 1
    logging.info("Ending self_play()")

def play(player, opponent):
    """ Game between two players, for evaluation """

    # Create the evaluation match queue of processes
    logging.info("Staring to evaluate")
    results = []
    for game_id in trange(EVAL_MATCHS):
        results.append(pickle.loads(Game(player, game_id, opponent=deepcopy(opponent)).__call__()))
    return results
