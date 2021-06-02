import logging
from utils.config import *
from model.mcts_pure import MCTS
from lib.gobang import GobangEnv
from utils.utils import _prepare_state


class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, color="black", gobang_size=GOBANG_SIZE, opponent=False):
        self.gobang_size = gobang_size
        self.board = self._create_board(color)
        self.player_color = 2 if color == "black" else 1
        self.player = MCTS()
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

    def _play(self):
        """ Choose a move depending on MCTS or not """

        action = self.player.search(self.board)

        state, reward, done = self.board.step(action)
        logging.warning("move:\n{}".format(self.board.board.board))
        if done:
            logging.debug("Finish a play: {}".format(self.board.board.board))
        return state, reward, done, action

    def run(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated move. Also returns the winner in order to create the
        training dataset
        """
        done = False
        state = self.board.reset()
        moves = 0
        logging.debug("start one match")
        while not done:
            # Prevent game from cycling
            if not moves < MOVE_LIMIT:
                return

            # For evaluation
            if self.opponent:
                state, reward, done, action = self._play()
                state, reward, done, action = self._play()
                moves += 2

            # For self-play
            else:
                logging.debug("Starting self-play")
                state = _prepare_state(state)
                new_state, reward, done, action = self._play()
                self._swap_color()
                state = new_state
                moves += 1

        logging.info("Finish one match. moves:{}".format(moves))

    def solo_play(self, move=None):
        """ Used to play against a human or for GTP, cant be called
        in a multiprocess scenario """

        ## Agent plays the first move of the game
        if move is None:
            state = _prepare_state(self.board.state)
            state, reward, done, move = self._play()
            self._swap_color()
            return move, done
        ## Otherwise just play a move and answer it
        else:
            state, reward, done = self.board.step(move)
            self.player.advance(move)
            self._swap_color()
            return True, done

    def reset(self):
        state = self.board.reset()


def play_with_pure_mcts():
    game = Game(opponent=MCTS())
    game.run()

if __name__ == '__main__':
    play_with_pure_mcts()