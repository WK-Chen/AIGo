import logging
import numpy as np
import pickle
from utils.config import *
from model.mcts import MCTS
from .go import GobangEnv
from utils.utils import _prepare_state


class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, player, id, color="black", mcts_flag=MCTS_FLAG, gobang_size=GOBANG_SIZE, opponent=False):
        self.gobang_size = gobang_size
        self.id = id + 1
        self.human_pass = False
        self.board = self._create_board(color)
        self.player_color = 2 if color == "black" else 1
        self.mcts = mcts_flag
        if mcts_flag:
            self.mcts = MCTS()
        self.player = player
        self.opponent = opponent

    def _create_board(self, color):
        """
        Create a board with a gobang_size and the color is for the starting player
        """
        board = GobangEnv(color, self.gobang_size)
        board.reset()
        return board

    def _swap_color(self):
        if self.player_color == 1:
            self.player_color = 2
        else:
            self.player_color = 1

    def _get_move(self, board, probs):
        """ Select a move without MCTS """
        player_move = None
        legal_moves = board.get_legal_moves()

        while player_move not in legal_moves and len(legal_moves) > 0:
            player_move = np.random.choice(probs.shape[0], p=probs)
            if player_move not in legal_moves:
                old_prob = probs[player_move]
                probs = probs + (old_prob / (probs.shape[0] - 1))
                probs[player_move] = 0

        return player_move

    def _play(self, state, player, competitive=False):
        """ Choose a move depending on MCTS or not """

        if self.mcts:
            action_scores, action = self.mcts.search(self.board, player, competitive=competitive)
            logging.debug("A mcts search has finished")

        else:
            feature_maps = player.extractor(state)
            _, probs = player.predict(state)
            probs = probs[0].cpu().data.numpy()

            action = self._get_move(self.board, probs)

            action_scores = np.zeros((self.gobang_size ** 2))
            action_scores[action] = 1

        state, reward, done = self.board.step(action)
        if done:
            logging.debug("Finish a play: {}".format(self.board.board.board))
        return state, reward, done, action_scores, action

    def __call__(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated move. Also returns the winner in order to create the
        training dataset
        """
        done = False
        state = self.board.reset()
        dataset = []
        moves = 0
        comp = False
        while not done:
            ## Prevent game from cycling
            if moves > MOVE_LIMIT:
                reward = 0
                if self.opponent:
                    logging.warning("[EVALUATION] %d :Reach move limit, black lose" % self.id)
                    return pickle.dumps([reward])
                return pickle.dumps((dataset, reward))

            ## Adaptative temperature to stop exploration
            if moves > TEMPERATURE_MOVE:
                comp = True

            # For evaluation
            if self.opponent:
                state, reward, done, _, action = self._play(
                    _prepare_state(state), self.player, competitive=True)
                state, reward, done, _, action = self._play(
                    _prepare_state(state), self.opponent, competitive=True)
                moves += 2

            # For self-play
            else:
                logging.debug("Starting self-play")
                state = _prepare_state(state)
                new_state, reward, done, probs, action = self._play(
                    state, self.player, competitive=comp)
                self._swap_color()
                # logging.info("Move {}, Swap color".format(moves))
                dataset.append((state.cpu().data.numpy(), probs, self.player_color, action))
                state = new_state
                moves += 1

        logging.debug("Finish one match")
        # Pickle the result because multiprocessing
        if self.opponent:
            logging.info("[EVALUATION] Match %d done in eval after %d moves, winner %s"
                         % (self.id, moves, "black" if reward == 0 else "white"))
            return pickle.dumps([reward])
        # print(dataset, reward)
        return pickle.dumps((dataset, reward))

    def solo_play(self, move=None):
        """ Used to play against a human or for GTP, cant be called
        in a multiprocess scenario """

        ## Agent plays the first move of the game
        if move is None:
            state = _prepare_state(self.board.state)
            state, reward, done, probs, move = self._play(state, self.player, self.human_pass, competitive=True)
            self._swap_color()
            return move
        ## Otherwise just play a move and answer it
        else:
            state, reward, done = self.board.step(move)
            if move != self.board.board_size ** 2:
                self.mcts.advance(move)
            else:
                self.human_pass = True
            self._swap_color()
            return True

    def reset(self):
        state = self.board.reset()
