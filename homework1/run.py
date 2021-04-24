from typing import List
import pygame
import datetime
import argparse

from utils import read_file, set_matrix, random_matrix,set_goal
from astar import A_start

WHITE = (255, 255, 255)


def init(matrix):
    # pygame初始化
    pygame.init()
    Game_font = pygame.font.SysFont("arial", 32)
    screen = pygame.display.set_mode((300, 300), 0, 32)
    screen.fill(WHITE)
    refresh(screen, matrix)
    pygame.display.update()
    return screen


def convert_int_to_image(num: int):
    return pygame.image.load('./source/{}.png'.format(num))


def refresh(screen, matrixs: List) -> None:
    # screen.fill((255, 255, 255))
    for i, rows in enumerate(matrixs):
        for j, num in enumerate(rows):
            screen.blit(convert_int_to_image(num), [j * 50, i * 50])
    pygame.display.update()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--N", default=None, type=int, help="Input row length."
    )
    parser.add_argument(
        "--file_path", default=None, type=str, help="Input example path."
    )
    args = parser.parse_args()

    row = args.N
    if args.file_path:
        matrix = set_matrix(read_file(args.file_path), row)
    else:
        matrix = random_matrix(row)
    goal = set_goal(row)

    start_t = datetime.datetime.now()
    stack = A_start(matrix, goal, time_limit=60)
    end_t = datetime.datetime.now()
    print("time = {}s".format((end_t - start_t).total_seconds()))

    # visualize
    screen = init(matrix)
    step = 0
    if not stack:
        print("failed!")
        return
    while stack:
        step += 1
        new_state = stack.pop()
        refresh(screen, new_state)
        pygame.time.wait(1000)
    pygame.time.wait(2000)
    print("step = {}".format(step))


if __name__ == "__main__":
    main()
