"""Logger utils"""
import logging
import time


class Timer:
    """
    A simple timer.
    """
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        """
        Get the start time.
        """
        self.start_time = time.time()

    def toc(self, average=True):
        """
        Compute duration of the period
        """
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        """
        Clear values.
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


def get_logger(name='root'):
    """
    Get Logger.
    """
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logg = logging.getLogger(name)
    logg.setLevel(logging.DEBUG)
    logg.addHandler(handler)

    return logg


logger = get_logger('root')