import multiprocessing
import signal
import time
from lib.process import MyPool

def a(locka, lockb):
    while True:
        print("acquire a")
        locka.acquire()
        print("a")
        print("release b")
        lockb.release()
        time.sleep(0.5)
def b(locka, lockb):
    while True:
        print("acquire b")
        lockb.acquire()
        print("b")
        print("release a")
        locka.release()
        time.sleep(0.5)

def main():
    multiprocessing.set_start_method('spawn', force=True)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(1)
    locka = multiprocessing.Manager().Lock()
    lockb = multiprocessing.Manager().Lock()
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        self_play_proc = pool.apply_async(a, args=(locka, lockb))
        train_proc = pool.apply_async(b, args=(locka, lockb))
        # self_play_proc.get(60000)
        # train_proc.get(60000)

    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
