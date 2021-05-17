import pygame as pygame
from rule import Rule
from utils.utils import load_player
from lib.game import Game
from utils.config import *
from time import sleep

def main(round):
    player, checkpoint = load_player(round)
    ai = Game(player, 0, opponent=None)
    print("model load finished")
    pygame.init()
    size = width, height = 313, 313
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('gobang')
    font = pygame.font.SysFont('arial', 36)
    clock = pygame.time.Clock()
    game_over = False
    rule = Rule()
    clock.tick(20)
    screen.blit(rule.chessboard, (0, 0))
    pygame.display.update()


    while True:
        # 设置帧率
        if game_over:
            sleep(3)
            break
        for event in pygame.event.get():
            # 退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if rule.black_turn:
                move, game_over = ai.solo_play()
                i, j = move % rule.lines, move // rule.lines
                drop_success = rule.drop_at(i, j)
                if drop_success:
                    rule.swap_color()
            else:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    i, j = rule.get_coord(event.pos)
                    drop_success = rule.drop_at(i, j)
                    _, game_over = ai.solo_play(j*GOBANG_SIZE+i)
                    if drop_success:
                        rule.swap_color()
            screen.blit(rule.chessboard, (0, 0))
            pygame.display.update()
            if game_over:
                text = "{} wins!"
                win_text = font.render(text, True, (0, 0, 0))
                rule.chessboard.blit(win_text, (0,0))





if __name__ == '__main__':
    main(round=19)
