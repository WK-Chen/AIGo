import multiprocessing
import time
import signal
import os
from lib.play import play, self_play
from train import train
from lib.process import MyPool
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')


def main(round):
    while True:
        logging.info("Starting Round: {}".format(round))
        # self_play(round)
        train(round)
        logging.info("Finished Round: {}".format(round))
        round += 1
        break


if __name__ == "__main__":
    main(round=0)
