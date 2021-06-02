import multiprocessing
from lib.train import train
from lib.play import self_play
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')


def main(round):
    logging.info("Starting Round: {}".format(round))
    #writer = SummaryWriter()
    # self_play(round, 1)
    train(round, 1)
    logging.info("Finished Round: {}".format(round))
    # round += 1


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main(0)
