import pygame as pygame
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_Alphazero import MCTSPlayer
from net import PolicyValueNet
from config import *
from GUI import GUI
import time


class Human(object):
    """
    human player
    """

    def __init__(self, UI):
        self.player = None
        self.left = UI.left
        self.space = UI.space
        self.top = UI.top
        self.gobang_size = BOARD_WIDTH

    def set_player_ind(self, p):
        self.player = p

    def get_move(self, pos):
        x, y = pos
        i, j = (0, 0)
        oppo_x = x - self.left
        if oppo_x > 0:
            i = round(oppo_x / self.space)
        oppo_y = y - self.top
        if oppo_y > 0:
            j = round(oppo_y / self.space)
        return i * self.gobang_size + j

    def get_action(self, board):
        sensible_moves = board.availables
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    move = self.get_move(event.pos)
                    if move not in sensible_moves:
                        print("illegal")
                        pass
                    else:
                        return move

    def __str__(self):
        return "Human {}".format(self.player)


def start_play(UI, game, ai, human, ai_first=True):
    """start a game between two players"""
    UI.init()
    start_player = 0 if ai_first else 1
    game.board.init_board(start_player)
    p1, p2 = game.board.players
    ai.set_player_ind(p1)
    human.set_player_ind(p2)
    players = {p1: ai, p2: human}

    while True:
        current_player = game.board.get_current_player()
        player_in_turn = players[current_player]
        move = player_in_turn.get_action(game.board)
        game.board.do_move(move)
        UI.update(move // BOARD_WIDTH, move % BOARD_WIDTH, current_player)
        end, winner = game.board.game_end()
        if end:
            if winner != -1:
                print("Game end. Winner is", players[winner])
            else:
                print("Game end. Tie")
            time.sleep(5)
            return

def run():
    model_path = "./saved_models/game_id_2200_win_pure_mcts_3000.model"
    width = BOARD_WIDTH
    height = BOARD_WIDTH
    board = Board(width=width, height=height, n_in_row=N_IN_ROW)
    game = Game(board)
    best_policy = PolicyValueNet(width, height, model_file=model_path)
    ai = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
    UI = GUI()
    human = Human(UI)
    start_play(UI, game, ai, human)


if __name__ == '__main__':
    run()
