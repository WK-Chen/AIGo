import logging
import numpy as np
import pickle
import os
from time import sleep
import timeit
import re
from tqdm import trange
from copy import deepcopy
from utils.config import *
from utils.utils import get_player, load_player
from lib.process import create_matches, evaluate_matches
from lib.game import Game
from tensorboardX import SummaryWriter


def self_play(round, target_round):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    logging.info("Start self_play()")
    if not round:
        game_id = 0
    else:
        list = os.listdir('./data/')
        list.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        game_id = int(list[-1]) + 1
    logging.info("Continue from game_id:{}".format(game_id))
    writer = SummaryWriter()
    while game_id < target_round * SIMULATION_PER_ROUND:
        # Load the player when restarting training
        logging.info("Loading Player")

        if round != 0:
            player, _ = load_player(None, last_player=True, best_player=False)
        else:
            player = get_player()

        # Create the self-play match
        logging.info("Creating matches ...")
        queue, results = create_matches(player, opponent=None, match_number=SIMULATION_PER_ROUND, cores=MCTS_PARALLEL)
        start_time = timeit.default_timer()
        moves = []
        try:
            queue.join()
            for _ in range(SIMULATION_PER_ROUND):
                result = results.get()
                if result:
                    with open("data/{}".format(game_id), 'wb') as f:
                        f.write(result[1])
                    moves.append(result[0])
                    game_id += 1

                if game_id % 5 == 0:
                    writer.add_scalar("scalar/match_moves", np.mean(moves), game_id)
                    moves = []
            final_time = timeit.default_timer() - start_time
            logging.info("Done fetching in %.3f seconds, average duration:"
                         " %.3f seconds" % (final_time, final_time / SIMULATION_PER_ROUND))
        finally:
            queue.close()
            results.close()

        sleep(60)
        round += 1
    writer.close()
    logging.info("Ending self_play()")


def play(player, opponent):
    """ Game between two players, for evaluation """

    # Create the evaluation match queue of processes
    logging.info("Staring to evaluate")
    queue, results = evaluate_matches(player, opponent=opponent, evalate_number=EVAL_MATCHS, cores=4)
    wins = []
    try:
        queue.join()
        for _ in range(EVAL_MATCHS):
            result = results.get()
            if result:
                wins.append(pickle.loads(result))
    finally:
        queue.close()
        results.close()
    return wins

