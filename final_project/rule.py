import sys
import random
import pygame
from collections import namedtuple
from utils.config import *

# 具名元组声明
Position = namedtuple('Position', ['x', 'y'])


class Rule(object):
    def __init__(self, gobang_size=GOBANG_SIZE):
        # 图片资源命名
        self.background_filename = 'img/chessboard.png'
        self.whiteBall_filename = 'img/whiteBall.png'
        self.blackBall_filename = 'img/blackBall.png'
        if gobang_size == 15:
            self.top, self.left, self.space, self.lines = (16, 16, 20, 15)
        elif gobang_size == 9:
            self.top, self.left, self.space, self.lines = (77, 77, 20, 9)
        else:
            print("Wrong size")
        # 棋盘格子线颜色(黑)
        self.color = (0, 0, 0)

        self.black_turn = True  # 黑子先手
        self.ball_coord = []  # 记录黑子和白子逻辑位置

        # 图片资源加载
        self.chessboard = pygame.image.load(self.background_filename)
        # 保留透明底
        self.whiteBall = pygame.image.load(self.whiteBall_filename).convert_alpha()
        self.blackBall = pygame.image.load(self.blackBall_filename).convert_alpha()
        self.font = pygame.font.SysFont('arial', 16)
        self.ball_rect = self.whiteBall.get_rect()
        # 初始化点位
        self.points = [[] for _ in range(self.lines)]
        for i in range(self.lines):
            for j in range(self.lines):
                self.points[i].append(Position(self.left + i * self.space, self.top + j * self.space))

    # 在(i,j)位置落子
    def drop_at(self, i, j):
        if not self.check_at(i, j):
            return False
        pos_x = self.points[i][j].x - int(self.ball_rect.width / 2)
        pos_y = self.points[i][j].y - int(self.ball_rect.height / 2)

        ball_pos = {'type': 0 if self.black_turn else 1, 'coord': Position(i, j)}
        if self.black_turn:  # 轮到黑子下
            self.chessboard.blit(self.blackBall, (pos_x, pos_y))
        else:
            self.chessboard.blit(self.whiteBall, (pos_x, pos_y))

        self.ball_coord.append(ball_pos)  # 记录已落子信息
        return True

    # 判断是否已产生胜方
    def check_over(self):
        if len(self.ball_coord) > 8:  # 只有黑白子已下4枚以上才判断
            direct = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、斜、反斜 四个方向检查
            for d in direct:
                if self._check_direct(d):
                    return True
        return False

    # 判断最后一个棋子某个方向是否连成5子，direct:(1,0),(0,1),(1,1),(1,-1)
    def _check_direct(self, direct):
        dt_x, dt_y = direct
        last = self.ball_coord[-1]
        line_ball = []  # 存放在一条线上的棋子
        for ball in self.ball_coord:
            if ball['type'] == last['type']:
                x = ball['coord'].x - last['coord'].x
                y = ball['coord'].y - last['coord'].y
                if dt_x == 0:
                    if x == 0:
                        line_ball.append(ball['coord'])
                        continue
                if dt_y == 0:
                    if y == 0:
                        line_ball.append(ball['coord'])
                        continue
                if x * dt_y == y * dt_x:
                    line_ball.append(ball['coord'])

        if len(line_ball) >= 5:  # 只有5子及以上才继续判断
            sorted_line = sorted(line_ball)
            for i, item in enumerate(sorted_line):
                index = i + 4
                if index < len(sorted_line):
                    if dt_x == 0:
                        y1 = item.y
                        y2 = sorted_line[index].y
                        if abs(y1 - y2) == 4:  # 此点和第5个点比较y值，如相差为4则连成5子
                            return True
                    else:
                        x1 = item.x
                        x2 = sorted_line[index].x
                        if abs(x1 - x2) == 4:  # 此点和第5个点比较x值，如相差为4则连成5子
                            return True
                else:
                    break
        return False

    # 检查(i,j)位置是否已占用
    def check_at(self, i, j):
        for item in self.ball_coord:
            if (i, j) == item['coord']:
                return False
        return True

    # 通过物理坐标获取逻辑坐标
    def get_coord(self, pos):
        x, y = pos
        i, j = (0, 0)
        oppo_x = x - self.left
        if oppo_x > 0:
            i = round(oppo_x / self.space)  # 四舍五入取整
        oppo_y = y - self.top
        if oppo_y > 0:
            j = round(oppo_y / self.space)
        return i, j

    def get_test_data(self):
        # 测试用数组
        test_data = [0, 1, 2, 4, 6, 32, 48, 77, 79, 80]
        if test_data:
            cur = random.sample(test_data, 1).pop(0)
            test_data.remove(cur)
            return cur
        return -1

    def swap_color(self):
        self.black_turn = not self.black_turn
