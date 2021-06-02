import multiprocessing
from lib.play import self_play
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')


def main(round):
    logging.info("Starting Round: {}".format(round))
    self_play(round, 1)
    logging.info("Finished Round: {}".format(round))


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main(0)
