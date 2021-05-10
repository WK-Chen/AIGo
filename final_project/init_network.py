import time
import torch
from model.agent import Player
from utils.config import *

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

def init():
    player = Player()
    optimizer = create_optimizer(player, LR)
    state = {
        'version': False,
        'lr': LR,
        'total_ite': 0,
        'optimizer': optimizer.state_dict()
    }
    player.save_models(state, "RAW")

if __name__ == '__main__':
    init()