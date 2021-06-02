import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import random
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool
import pickle
import timeit
from collections import defaultdict, deque
from tensorboardX import SummaryWriter
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_Alphazero import MCTSPlayer
from net import PolicyValueNet
from config import *
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')

class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_WIDTH
        self.n_in_row = N_IN_ROW
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = LEARN_RATE
        self.lr_multiplier = LR_MULTIPLIER
        self.temp = TEMP
        self.n_playout = N_PLAYOUT
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = PLAY_BATCH_SIZE
        self.epochs = EPOCHS
        self.kl_targ = KL_TARG
        self.check_freq = CHECK_FREQ
        self.game_batch_num = GAME_BATCH_NUM
        self.best_win_ratio = BEST_WIN_RATIO
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = PURE_MCTS_PLAYOUT_NUM
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, writer, game_id, n_games=1):
        """collect self-play data for training"""
        results = []
        episode_len = []
        start_time = timeit.default_timer()
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            episode_len.append(len(play_data))
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            game_id += 1
        final_time = timeit.default_timer() - start_time
        logging.info("Done fetching in %.3f seconds, average duration:"
                     " %.3f seconds" % (final_time, final_time / n_games))
        logging.info(("episode_len:{}".format(np.array(episode_len).mean())))
        writer.add_scalar("scalar/episode_len", float(np.array(episode_len).mean()), game_id)
        return game_id

    def collect_selfplay_data_parallel(self, writer, game_id, n_games=1):
        """collect self-play data for training"""
        pool = Pool(SELF_PLAY_PARALLEL)
        results = []
        episode_len = []
        for _ in range(n_games):
            result = pool.apply_async(self.game.start_self_play, args=(self.mcts_player, 0, self.temp))
            results.append(result)
        start_time = timeit.default_timer()
        try:
            pool.close()
            pool.join()
            for result in results:
                winner, play_data = result.get()
                # with open("data/{}".format(game_id), 'wb') as f:
                #     f.write(pickle.dumps(winner, zip(*play_data)))
                play_data = list(play_data)[:]
                episode_len.append(len(play_data))
                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
                game_id += 1
            final_time = timeit.default_timer() - start_time
            logging.info("Done fetching in %.3f seconds, average duration:"
                         " %.3f seconds" % (final_time, final_time / n_games))
            logging.info(("episode_len:{}".format(np.array(episode_len).mean())))
            writer.add_scalar("scalar/episode_len", float(np.array(episode_len).mean()), game_id)
        except Exception as e:
            logging.error(e)
        finally:
            return game_id

    def policy_update(self, writer, total_ite):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        losses = []
        entropys= []
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            losses.append(loss)
            entropys.append(entropy)

            total_ite += 1
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        writer.add_scalar("scalar/loss", np.mean(losses), total_ite)
        writer.add_scalar("scalar/policy_entropy", np.mean(entropys), total_ite)
        logging.info("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},".format(
            kl, self.lr_multiplier, np.mean(losses), np.mean(entropys)))
        return np.mean(losses), np.mean(entropys), total_ite

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        writer = SummaryWriter()
        game_id = 0
        total_ite = 0
        try:
            while game_id < self.game_batch_num:
                # logging.info("begin self play")
                game_id = self.collect_selfplay_data_parallel(writer, game_id, self.play_batch_size)
                # logging.info("start training")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, total_ite = self.policy_update(writer, total_ite)
                # check the performance of the current model, and save the model params
                if game_id % self.check_freq == 0:
                    logging.info("current self-play batch: {}".format(game_id))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./saved_models/{}.model'.format(game_id))
                    self.policy_value_net.save_model('./saved_models/current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        logging.info("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(
                            './saved_models/game_id_{}_win_pure_mcts_{}.model'.
                                format(game_id, self.pure_mcts_playout_num))
                        self.policy_value_net.save_model('./saved_models/best_policy_6_4.model')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = BEST_WIN_RATIO
                logging.info("game round {} finished !".format(game_id))
            logging.info("finish")
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    training_pipeline = TrainPipeline()
    training_pipeline.run()
