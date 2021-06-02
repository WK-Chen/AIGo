import numpy as np
import random
import os
import pickle
import time
from lib.dataset import SelfPlayDataset
from lib.evaluate import evaluate
from utils.utils import load_player
from copy import deepcopy
import logging
from torch.utils.data import DataLoader
from utils.config import *
from model.agent import Player
from tqdm import tqdm


class AlphaLoss(torch.nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probs
    p : probs

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, winner, self_play_winner, probs, self_play_probs):
        value_error = (self_play_winner - winner) ** 2
        policy_error = torch.sum((-self_play_probs * (1e-6 + probs).log()), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
        return total_error


def fetch_new_games(dataset, round, first_time=False):
    """ Update the dataset with new games from the databse """

    # Fetch new games in reverse order so we add the newest games first
    added_moves = 0
    added_games = 0
    #if round < 20 or first_time:
    if round < 20:
        new_games = ["./data/{}/".format(round) + game for game in os.listdir("./data/{}".format(round))]
    else:
        new_games = []
        for i in range(round, round - 20, -1):
            for game in os.listdir("./data/{}".format(i)):
                new_games.append("./data/{}/".format(i) + game)
    random.shuffle(new_games)

    i = 0
    # max_replacement = 1.0 if first_time else MAX_REPLACEMENT
    while added_moves < MOVES * 1.0:
        with open(new_games[i], 'rb') as f:
            number_moves = dataset.update(pickle.load(f))
        added_moves += number_moves
        added_games += 1
        i += 1

    logging.info("added games: {}, added moves: {}".format(added_games, added_moves))


def train_epoch(player, optimizer, example, criterion):
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()
    winner, probs = player.predict(example['state'])
    loss = criterion(winner, example['winner'], probs, example['move'])
    loss.backward()
    optimizer.step()

    return float(loss)


def update_lr(lr, optimizer, total_ite, lr_decay=LR_DECAY, lr_decay_ite=LR_DECAY_ITE):
    """ Decay learning rate by a factor of lr_decay every lr_decay_ite iteration """

    if total_ite % lr_decay_ite != 0 or lr <= 0.0001:
        return lr, optimizer

    logging.info("Decaying the learning rate !")
    lr = lr * lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, optimizer


def create_state(lr, total_ite, optimizer):
    """ Create a checkpoint to be saved """

    state = {
        'lr': lr,
        'total_ite': total_ite,
        'optimizer': optimizer.state_dict()
    }
    return state


def collate_fn(example):
    """ Custom way of collating example in dataloader """

    state = []
    probs = []
    winner = []
    for ex in example:
        state.extend(ex[0])
        probs.extend(ex[1])
        winner.extend(ex[2])

    state = torch.tensor(state, dtype=torch.float, device=DEVICE)
    probs = torch.tensor(probs, dtype=torch.float, device=DEVICE)
    winner = torch.tensor(winner, dtype=torch.float, device=DEVICE)
    return state, probs, winner


def create_optimizer(player, lr, param=None):
    """ Create or load a saved optimizer """

    joint_params = list(player.extractor.parameters()) + \
                   list(player.policy_net.parameters()) + \
                   list(player.value_net.parameters())

    if ADAM:
        opt = torch.optim.Adam(joint_params, lr=lr, weight_decay=L2_REG)
    else:
        opt = torch.optim.SGD(joint_params, lr=lr, weight_decay=L2_REG, momentum=MOMENTUM)

    if param:
        opt.load_state_dict(param)

    return opt


def train(round, target_round):
    """ Train the models using the data generated by the self-play """

    total_ite = 1
    lr = LR
    criterion = AlphaLoss()
    dataset = SelfPlayDataset()
    while not os.path.isdir("./data/{}".format(round)) or \
            len(os.listdir("./data/{}".format(round))) < SIMULATION_PER_ROUND:
        time.sleep(100)
    time.sleep(5)
    logging.info("Creating dataset ...")
    fetch_new_games(dataset, round, first_time=True)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

    while True:
        # First player either from disk or fresh
        if round != 0:
            logging.info("Loaded a player from round {}".format(round-1))
            player, checkpoint = load_player(round-1)
            optimizer = create_optimizer(player, lr, param=checkpoint['optimizer'])
            total_ite = checkpoint['total_ite']
            lr = checkpoint['lr']
        else:
            logging.info("Initializing a player")
            player = Player()
            optimizer = create_optimizer(player, lr)
            state = create_state(lr, total_ite, optimizer)
            player.save_models(state, round)
        best_player = deepcopy(player)

        logging.info("Starting to train !")

        batch_loss = []
        for epoch in range(2):
            for batch_idx, (state, move, winner) in enumerate(tqdm(dataloader)):
                lr, optimizer = update_lr(lr, optimizer, total_ite)

                # Evaluate a copy of the current network asynchronously
                example = {'state': state, 'winner': winner, 'move': move}
                loss = train_epoch(player, optimizer, example, criterion)
                batch_loss.append(loss)
                total_ite += 1
        logging.info("Averaged loss: {}".format(np.mean(batch_loss)))

        """
        pending_player = deepcopy(player)
        result = evaluate(pending_player, best_player)
        if result:
            best_player = pending_player
            logging.info("New version wins !")
        else:
            best_player = pending_player
            logging.info("New version lose !")
        state = create_state(lr, total_ite, optimizer)
        best_player.save_models(state, round + 1)
        """
        if round % 10 == 0:
            pending_player = deepcopy(player)
            result = evaluate(pending_player, best_player)
            if result:
                logging.info("New version wins !")
            else:
                logging.info("New version lose !")
        state = create_state(lr, total_ite, optimizer)
        player.save_models(state, round)
        round += 1
        if round >= target_round:
            break
        while not os.path.isdir("./data/{}".format(round)) or \
                len(os.listdir("./data/{}".format(round))) < SIMULATION_PER_ROUND:
            logging.info("train process sleeping")
            time.sleep(100)
        time.sleep(5)
        fetch_new_games(dataset, round)
