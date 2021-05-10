import pickle
import time
import timeit
from copy import deepcopy
from utils.config import *
from utils.utils import get_player, load_player
from .process import create_matches

# TODO change the loaded_version and model_path
def self_play(model_path, loaded_version):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    game_id = 0
    current_version = 1
    player = False
    print("[INFO] Start self_play()")
    while True:
        # Load the player when restarting traning
        print("[INFO] Loading player")
        if loaded_version:
            new_player, checkpoint = load_player(model_path, loaded_version)
            current_version = checkpoint['version'] + 1
            loaded_version = False
        else:
            new_player, checkpoint = get_player(model_path, current_version)
            if new_player:
                current_version = checkpoint['version'] + 1
        print("[INFO] Load player finished")

        # Waiting for the first player to be saved
        print("[PLAY] Current improvement level: %d" % current_version)
        if current_version == 1 and not player and not new_player:
            print("[PLAY] Waiting for first player")
            time.sleep(5)
            continue

        if new_player:
            player = new_player
            print("[PLAY] New player !")

        ## Create the self-play match queue of processes
        queue, results = create_matches(player, cores=PARALLEL_SELF_PLAY, match_number=SELF_PLAY_MATCH)
        print("[RESULT] queue:{} \n results:{}".format(queue, results))
        print("[PLAY] Starting to fetch fresh games")
        start_time = timeit.default_timer()

        try:
            queue.join()

            ## Collect the results and push them in the database
            for _ in range(SELF_PLAY_MATCH):
                result = results.get()
                if result:
                    print("[WARNING] Writing data")
                    with open("data/id_{}".format(game_id), 'wb') as f:
                        pickle.dump(result, f)
                    game_id += 1
            final_time = timeit.default_timer() - start_time
            print("[PLAY] Done fetching in %.3f seconds, average duration:"
                  " %.3f seconds" % ((final_time, final_time / SELF_PLAY_MATCH)))
        finally:
            queue.close()
            results.close()


def play(player, opponent):
    """ Game between two players, for evaluation """

    ## Create the evaluation match queue of processes
    queue, results = create_matches(deepcopy(player), opponent=deepcopy(opponent), \
                                    cores=PARALLEL_EVAL, match_number=EVAL_MATCHS)
    try:
        queue.join()

        ## Gather the results and push them into a result queue
        ## that will be sent back to the evaluation process
        print("[EVALUATION] Starting to fetch fresh games")
        final_result = []
        for idx in range(EVAL_MATCHS):
            result = results.get()
            if result:
                final_result.append(pickle.loads(result))
        print("[EVALUATION] Done fetching")
    finally:
        queue.close()
        results.close()
    return final_result