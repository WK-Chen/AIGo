from net import PolicyValueNet
from mcts_Alphazero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from collections import defaultdict
from config import *
from game import Board, Game
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')

def evaluate(model_path, pure_mcts_playout_num, n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    """
    net = PolicyValueNet(BOARD_WIDTH, BOARD_WIDTH, model_file=model_path)
    current_mcts_player = MCTSPlayer(net.policy_value_fn, c_puct=5, n_playout=400)
    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    game = Game(Board(width=BOARD_WIDTH,
                      height=BOARD_WIDTH,
                      n_in_row=N_IN_ROW))
    for i in range(n_games):
        winner = game.start_play(current_mcts_player,
                                 pure_mcts_player,
                                 start_player=i % 2,
                                 is_shown=0)
        win_cnt[winner] += 1
        logging.info("winner :{}".format(winner))
    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
        pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


if __name__ == '__main__':
    pure_mcts_playout_num = 1000
    for i in range(200, 4200, 200):
        logging.info("current model: {}".format(i))
        model_path = 'saved_models/{}.model'.format(i)
        win_ratio = evaluate(model_path, pure_mcts_playout_num)
        if win_ratio >= 0.9:
            pure_mcts_playout_num += 1000
