import json
import time
from pathlib import Path

import numpy as np
from torch import save
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cfg.config import config
from src.dataset import JointDataset
from src.log_utils import TensorBoardLog
from src.log_utils import logger
from src.model import init_train_model
from src.utils import collate_fn


def worker_init_fn(worker_id):
    """
    Initialize worker seed.

    Args:
        worker_id: Id of the current cpu worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    """
    Model training pipeline.
    Training starts only with the pretrained backbone (downloads automatically).
    """
    tensorboard_dir = Path(config.logs_dir, 'tensorboard_logs')
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize summary writer for tensorboard
    tb_logger = TensorBoardLog(str(tensorboard_dir))

    with Path(config.data_cfg_url).open('r') as file:
        data_paths = json.load(file)['train']

    dataset = JointDataset(
        root=config.dataset_root,
        paths=data_paths,
        img_size=config.img_size,
        augment=True,
        config=config,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )

    model, params = init_train_model(config, dataset.nid)

    optimizer = SGD(params, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * config.epochs), int(0.75 * config.epochs)], gamma=0.1)

    running_loss = []
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        for step, batch in enumerate(dataloader):
            batch = collate_fn(batch)

            # Warmup for the first steps
            if epoch == 1 and step <= config.warmup_steps:
                learning_rate = config.learning_rate * (step / config.warmup_steps) ** 4
                for param_lr in optimizer.param_groups:
                    param_lr['lr'] = learning_rate

            loss, log_loss = model(*batch)

            loss.backward()

            if (step + 1) % config.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % config.log_step == 0 and step != 0:
                log = f'epoch {epoch}/{config.epochs} ' \
                      f'step {step}/{len(dataloader)} ' \
                      f'loss {np.mean(running_loss)}'

                logger.info(log)

                running_loss = []
            else:
                running_loss.append(loss.detach().cpu().numpy())

            global_step = (epoch - 1) * len(dataloader) + step
            tb_logger.update(global_step, log_loss, optimizer)

        log = f'{epoch} epoch time {(time.time() - epoch_start) // 60} minutes'
        logger.info(log)

        checkpoint_name = str(Path(config.logs_dir, f'JDE-{epoch}.pt'))
        checkpoint = {
            'model': model.state_dict(),
        }

        save(checkpoint, checkpoint_name)

        scheduler.step()


if __name__ == "__main__":
    main()
