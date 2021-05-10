import multiprocessing
import time
import signal
import os
from lib.play import play, self_play
from lib.process import MyPool


def main(model_path, version):

    # Start method for PyTorch
    multiprocessing.set_start_method('spawn')

    # TODO to change
    if model_path is None:
        model_path = "new"

    # Catch SIGNINT
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        self_play_proc = pool.apply_async(self_play, args=(model_path, version,))
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()
    print("[INFO] finish!")

if __name__ == "__main__":
    main(model_path='GOBANG_SIZE_9', version=False)


