import argparse
import multiprocessing
from lib.play import self_play
from lib.train import train
import logging
import signal
from lib.process import MyPool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s',
    datefmt='%m-%d %H:%M:%S')


def main(round, target_round):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        self_play_proc = pool.apply_async(self_play, args=(round, target_round))
        train_proc = pool.apply_async(train, args=(round, target_round))
        self_play_proc.get(60000)
        train_proc.get(60000)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()
    logging.info("finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--round", type=int, required=True, help="Input start round."
    )
    parser.add_argument(
        "--target_round", type=int, required=True, help="Input target round."
    )
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(round=args.round, target_round=args.target_round)
