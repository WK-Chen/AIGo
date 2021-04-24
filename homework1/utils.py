from typing import List
import random

def read_file(path='./source/example.txt'):
    with open(path, 'r') as f:
        line = f.readline().split(' ')
        line = [int(num) for num in line]
    return line


def set_matrix(line, row) -> List[List[int]]:
    return [line[4 * i:4 * (i + 1)] for i in range(row)]


def random_matrix(row) -> List[List[int]]:
    upper = row*row
    line = [i for i in range(upper)]
    for _ in range(100):
        a = random.randint(0, upper-1)
        b = random.randint(0, upper-1)
        line[a], line[b] = line[b], line[a]

    return [line[4 * i:4 * (i + 1)] for i in range(row)]


def set_goal(row=4):
    goal = [i + 1 for i in range(row * row)]
    goal = [goal[4 * i:4 * (i + 1)] for i in range(row)]
    goal[-1][-1] = 0
    return goal


def manhattan_dis(cur_node, end_node):
    '''
    计算曼哈顿距离
    :param cur_state: 当前状态
    :return: 到目的状态的曼哈顿距离
    '''
    cur_state = cur_node.state
    end_state = end_node.state
    dist = 0
    N = len(cur_state)
    for i in range(N):
        for j in range(N):
            if cur_state[i][j] == end_state[i][j]:
                continue
            num = cur_state[i][j]
            if num == 0:
                x = N - 1
                y = N - 1
            else:
                x = num / N  # 理论横坐标
                y = num - N * x - 1  # 理论的纵坐标
            dist += (abs(x - i) + abs(y - j))

    return dist
