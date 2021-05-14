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


def self_play(round):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    logging.info("Start self_play()")
    # Load the player when restarting training
    logging.info("Loading Player")
    if round != 0:
        player, _ = load_player(round)
    else:
        player = get_player()

    logging.info("Loaded a player from round {}".format(round))

    if not player:
        logging.error("Player didn't load correctly!")
        return
    if not os.path.isdir("data/{}".format(round)):
        os.mkdir("data/{}".format(round))
    else:
        logging.error("A directory already exists!")
    # Create the self-play match
    logging.info("Creating a match")

    results = []
    count = 0
    start_time = timeit.default_timer()
    for game_id in trange(SIMULATION_PER_ROUND):
        results.append(Game(player, game_id, opponent=None).__call__())
        if (game_id + 1) % 10 == 0:
            logging.info("Collecting data")
            for id, result in enumerate(results):
                with open("data/{}/id_{}".format(round, count*10+id), 'wb') as f:
                    f.write(result)
            results = []
            count += 1
            logging.info("Data collected")
    final_time = timeit.default_timer() - start_time
    logging.info("Done saving in %.3f seconds, average duration:"
                 " %.3f seconds" % (final_time, final_time / SIMULATION_PER_ROUND))



def play(player, opponent):
    """ Game between two players, for evaluation """

    # Create the evaluation match queue of processes
    logging.info("Staring to evaluate")
    queue, results = create_matches(deepcopy(player), opponent=deepcopy(opponent),
                                    cores=PARALLEL_EVAL, match_number=EVAL_MATCHS)
    try:
        queue.join()

        # Gather the results and push them into a result queue
        # that will be sent back to the evaluation process
        logging.info("Starting to fetch fresh games")
        final_result = []
        for idx in range(EVAL_MATCHS):
            result = results.get()
            if result:
                final_result.append(pickle.loads(result))
        logging.info("Done fetching")
    finally:
        queue.close()
        results.close()
    return final_result
