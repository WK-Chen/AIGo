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


def main(model_path):
    # Start method for PyTorch
    multiprocessing.set_start_method('spawn')

    # Catch SIGNINT
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)
    logging.info("Starting")
    try:
        self_play_proc = pool.apply_async(self_play, args=(model_path, 0))

        train_proc = pool.apply_async(train, args=(model_path, 0, './data'))

        ## Comment one line or the other to get the stack trace
        ## Must add a loooooong timer otherwise signals are not caught
        # self_play_proc.get(60000000)
        train_proc.get(60000000)

    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()
        logging.info("Pool finished")
    logging.info("Finished!")

if __name__ == "__main__":
    main(model_path=None, )


