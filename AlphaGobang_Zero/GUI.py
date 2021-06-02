import random
import pygame
from collections import namedtuple
from config import *

# 具名元组声明
Position = namedtuple('Position', ['x', 'y'])


class GUI(object):
    def __init__(self, gobang_size=BOARD_WIDTH):
        # 图片资源命名
        self.background_filename = "./img/chessboard.png"
        self.whiteBall_filename = './img/whiteBall.png'
        self.blackBall_filename = './img/blackBall.png'
        # restrict to board size 9
        self.gobang_size = gobang_size
        self.top, self.left, self.space, self.lines = (30, 30, 40, 9)

    def init(self):
        pygame.init()
        size = 374, 384
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('Gobang')
        font = pygame.font.SysFont('arial', 36)
        clock = pygame.time.Clock()
        clock.tick(20)
        # 图片资源加载
        self.chessboard = pygame.image.load(self.background_filename)
        self.whiteBall = pygame.image.load(self.whiteBall_filename).convert_alpha()
        self.blackBall = pygame.image.load(self.blackBall_filename).convert_alpha()
        self.font = pygame.font.SysFont('arial', 16)
        self.ball_rect = self.whiteBall.get_rect()
        # 初始化点位
        self.points = [[] for _ in range(self.lines)]
        for i in range(self.lines):
            for j in range(self.lines):
                self.points[i].append(Position(self.left + i * self.space, self.top + j * self.space))
        self.screen.blit(self.chessboard, (0, 0))
        pygame.display.update()

    # 在(i,j)位置落子
    def update(self, i, j, current_player):
        pos_x = self.points[i][j].x - int(self.ball_rect.width / 2)
        pos_y = self.points[i][j].y - int(self.ball_rect.height / 2)

        if current_player == 1:  # 轮到黑子下
            self.chessboard.blit(self.blackBall, (pos_x, pos_y))
        else:
            self.chessboard.blit(self.whiteBall, (pos_x, pos_y))

        self.screen.blit(self.chessboard, (0, 0))
        pygame.display.update()
