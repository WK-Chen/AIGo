import pygame as pygame
from rule import Rule


def main():
    # pygame初始化
    pygame.init()
    # 设置窗口大小
    size = width, height = 374, 384
    screen = pygame.display.set_mode(size)
    # 窗口标题
    pygame.display.set_caption('五子棋')
    # 游戏字体样式大小
    font = pygame.font.SysFont('arial', 36)
    # 设置时钟
    clock = pygame.time.Clock()
    # 游戏结束标志
    game_over = False
    # Rule()是核心类，实现落子及输赢判断等
    rule = Rule()
    # 初始化
    rule.__init__()

    while True:
        # 设置帧率
        clock.tick(20)
        for event in pygame.event.get():
            # 退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # 机器模拟测试
            cur = rule.get_test_data()

            # 测试数组为空时停止
            if cur == -1:
                continue

            line = rule.lines
            i, j = cur % line, int(cur / line)
            # 检查(i,j)位置能否被占用，如未被占用返回True
            if rule.check_at(i, j):
                # 在(i,j)位置落子，该函数将黑子或者白子画在棋盘上, 并切换黑白子
                rule.drop_at(i, j)

            # 以下是人机鼠标落位判定
            # # 落子事件
            # if event.type == pygame.MOUSEBUTTONDOWN and (not game_over):
            #     # 按下的是鼠标左键
            #     if event.button == 1:
            #         # 将物理坐标转换成矩阵的逻辑坐标
            #         i, j = rule.get_coord(event.pos)
            #         # 检查(i,j)位置能否被占用，如未被占用返回True
            #         if rule.check_at(i, j):
            #             # 在(i,j)位置落子，该函数将黑子或者白子画在棋盘上
            #             rule.drop_at(i, j)
            #
            #             # 检查是否存在五子连线，如存在则返回True
            #             if rule.check_over():
            #                 text = ''
            #                 # check_at会切换落子的顺序，所以轮到黑方落子，意味着最后落子方是白方，所以白方顺利
            #                 if rule.black_turn:
            #                     text = 'White side wins!'
            #                 else:
            #                     text = 'Black side wins!'
            #                 # 设置获胜语句
            #                 win_text = font.render(text, True, (0, 0, 0))
            #                 rule.chessboard.blit(win_text, (round(width / 2 - win_text.get_width() / 2),
            #                                                  round(height / 2 - win_text.get_height() / 2)))
            #                 game_over = True
            #         else:
            #             print('此位置已占用，不能在此落子')

        screen.blit(rule.chessboard, (0, 0))
        pygame.display.update()


if __name__ == '__main__': main()
