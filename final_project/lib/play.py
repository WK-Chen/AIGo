import logging
import pickle
import time
import timeit
from copy import deepcopy
from utils.config import *
from utils.utils import get_player, load_player
from .process import create_matches

def self_play(model_path, step):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    game_id = 0
    logging.info("Start self_play()")
    while True:
        # Load the player when restarting traning
        player = None
        logging.info("Loading Player")
        if model_path:
            logging.debug("model path is not none")
            player, _ = load_player(model_path, step)
            logging.info("Loaded a player from {}".format(model_path))
        else:
            logging.info("Initializing a player")
            player = get_player(model_path)

        logging.info("Load player finished")

        # Waiting for the first player to be saved
        if model_path:
            logging.info("Current player version is : {}".format(model_path[:5]))

        if not player:
            logging.error("Player didn't load correctly!")
            time.sleep(5)
            continue
        # print(player.extractor)
        # Create the self-play match queue of processes
        queue, results = create_matches(player, cores=PARALLEL_SELF_PLAY, match_number=SELF_PLAY_MATCH)

        start_time = timeit.default_timer()

        try:
            queue.join()
            # Collect the results and save them
            logging.info("Saving data")
            for _ in range(SELF_PLAY_MATCH):
                result = results.get()
                if result:
                    with open("data/id_{}".format(game_id), 'wb') as f:
                        f.write(result)
                    game_id += 1
            logging.info("Data saved")
            final_time = timeit.default_timer() - start_time
            logging.info("Done saving in %.3f seconds, average duration:"
                         " %.3f seconds" % (final_time, final_time / SELF_PLAY_MATCH))
        finally:
            queue.close()
            results.close()

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