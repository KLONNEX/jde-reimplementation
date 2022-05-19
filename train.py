import json
import time
from pathlib import Path

import numpy as np
from torch import save
from torch.nn import DataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cfg.config import config
from src.darknet import DarkNet
from src.darknet import ResidualBlock
from src.dataset import JointDataset
from src.log_utils import logger
from src.model import JDE
from src.model import YOLOv3
from src.model import load_darknet_weights
from src.utils import collate_fn


def worker_init_fn(worker_id):
    """
    Initialize worker seed.

    Args:
        worker_id: Id of the current cpu worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_train_model(cfg, nid):
    """
    Initialize model, and load weights into backbone.

    Args:
        cfg: Config parameters.
        nid (int): Number of unique identities in the dataset.

    Returns:
        network: Compiled train model with loss.
        optimizer_params (list): Trainable params of the model.
    """
    backbone = DarkNet(
        block=ResidualBlock,
        layer_nums=cfg.backbone_layers,
        in_channels=cfg.backbone_input_shape,
        out_channels=cfg.backbone_output_shape,
    )

    load_darknet_weights(model=backbone, weights=config.pretrained_path)

    net = YOLOv3(
        backbone=backbone,
        backbone_shape=cfg.backbone_output_shape,
        out_channel=cfg.out_channel,
    )

    network = JDE(
        extractor=net,
        config=cfg,
        nid=nid,
        ne=cfg.embedding_dim,
    )

    network.cuda().train()

    optimizer_params = []
    for param in network.parameters():
        if param.requires_grad:
            optimizer_params.append(param)

    network = DataParallel(network)

    return network, optimizer_params


def main():
    """
    Model training pipeline.
    Training starts only with the pretrained backbone (downloads automatically).
    """
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)

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

            loss = model(*batch)

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
