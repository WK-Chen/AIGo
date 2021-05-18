from lib.play import self_play
from lib.train import train
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')


def main(round):
    while round < 100:
        logging.info("Starting Round: {}".format(round))
        self_play(round)
        train(round)
        logging.info("Finished Round: {}".format(round))
        round += 1


if __name__ == "__main__":
    main(round=27)
