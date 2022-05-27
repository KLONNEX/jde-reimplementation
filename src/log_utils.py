"""Logger utils"""
import logging
import time

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLog:
    def __init__(self, log_dir):
        """
        Tensorboard logger for training metric visualization.

        Args:
            log_dir: Logging directory for tensorboard.
        """
        self.tb_logger = SummaryWriter(log_dir=log_dir, comment='Training writer.')

    def update(self, curr_step, log_dict, optimizer):
        """
        Save the current values into tensorboard logger.

        Args:
            curr_step: Current training step.
            log_dict: Dict with losses from the model.
            optimizer: Model optimizer to get the learning rate.
        """
        for key in log_dict.keys():
            self.tb_logger.add_scalar(
                tag=f'{key} loss',
                scalar_value=log_dict[key],
                global_step=curr_step,
            )

        for param_lr in optimizer.param_groups:
            self.tb_logger.add_scalar(
                tag='learning_rate',
                scalar_value=param_lr['lr'],
                global_step=curr_step,
            )


class Timer:
    """
    A simple timer.
    """
    def __init__(self):
        self.average_time = 0.
        self.total_time = 0.
        self.start_time = 0.
        self.duration = 0.
        self.calls = 0
        self.diff = 0.

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
        self.average_time = 0.
        self.total_time = 0.
        self.start_time = 0.
        self.duration = 0.
        self.calls = 0
        self.diff = 0.


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
logger.setLevel(logging.INFO)
