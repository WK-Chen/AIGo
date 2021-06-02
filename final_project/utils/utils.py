import os
import numpy as np
import random
from model.agent import Player
from utils.config import *
import logging


def _prepare_state(state):
    """
    Transform the numpy state into a PyTorch tensor with cuda if available
    """

    x = torch.from_numpy(np.array([state]))
    x = torch.as_tensor(x, dtype=torch.float, device=DEVICE)
    return x


def get_version(model_path, version):
    """ Either get the last versionration of
        the specific folder or verify it version exists """

    if int(version) == -1:
        files = os.listdir(model_path)
        if len(files) > 0:
            all_version = list(map(lambda x: int(x.split('-')[0]), files))
            all_version.sort()
            file_version = all_version[-1]
        else:
            return False
    else:
        test_file = "{}-extractor.pth.tar".format(version)
        if not os.path.isfile(os.path.join(folder_path, test_file)):
            return False
        file_version = version
    return file_version


def load_player(round):
    """ Load a player given a model_path """
    logging.info("load_player()")
    path = './saved_models/{}'.format(round)
    if not os.path.isdir(path):
        logging.error("Model path({}) incorrect !".format(path))
    player = Player()
    checkpoint = player.load_models(path)
    return player, checkpoint


def get_player():
    """ Initialize the model """
    player = Player()
    return player


def sample_rotation(state, num=1):
    """ Apply a certain number of random transformation to the input state """
    """Attention num=1 should not be changed, it may make errors"""
    # Create the dihedral group of a square with all the operations needed
    # in order to get the specific transformation and randomize their order
    dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
                ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
                (np.flipud, (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]
    random.shuffle(dh_group)

    states = []
    boards = (HISTORY + 1) * 2  ## Number of planes to rotate

    for idx in range(num):
        new_state = np.zeros((boards + 1, GOBANG_SIZE, GOBANG_SIZE,))
        new_state[:boards] = state[:boards]

        ## Apply the transformations in the tuple defining how to get
        ## the desired dihedral rotation / transformation
        for grp in dh_group[idx]:
            for i in range(boards):
                if isinstance(grp, tuple):
                    new_state[i] = grp[0](new_state[i], k=grp[1])
                elif grp is not None:
                    new_state[i] = grp(new_state[i])

        new_state[boards] = state[boards]
        states.append(new_state)

    if len(states) == 1:
        return np.array(states[0])
    return np.array(states)


def formate_state(state, prob, winner, rotation_num=ROTATION_NUM):
    dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
                ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
                (np.flipud, (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]
    random.shuffle(dh_group)

    states = []
    boards = (HISTORY + 1) * 2  # Number of planes to rotate
    probs = []

    for idx in range(rotation_num):
        new_state = np.zeros((boards + 1, GOBANG_SIZE, GOBANG_SIZE,))
        new_state[:boards] = state[:boards]
        new_prob = np.reshape(prob, (GOBANG_SIZE, GOBANG_SIZE))
        for grp in dh_group[idx]:
            if isinstance(grp, tuple):
                new_prob = grp[0](new_prob, k=grp[1])
            elif grp is not None:
                new_prob = grp(new_prob)
            for i in range(boards):
                if isinstance(grp, tuple):
                    new_state[i] = grp[0](new_state[i], k=grp[1])
                elif grp is not None:
                    new_state[i] = grp(new_state[i])

        new_state[boards] = state[boards]
        new_prob = np.reshape(new_prob, -1)
        states.append(new_state)
        probs.append(new_prob)
    winners = np.full((rotation_num, 1), winner)

    return np.array(states), np.array(probs), winners
