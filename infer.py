import os
from pathlib import Path

from cfg.config import config as default_config
from eval import eval_seq
from src.dataset import LoadVideo
from src.log_utils import logger
from src.model import init_eval_model
from src.utils import mkdir_if_missing


def track(opt):
    """
    Inference of the input video.
    Save the results into output-root (video, annotations and frames.).

    Args:
        opt: Config parameters.
    """
    result_root = Path(opt.output_root)
    result_name = result_root.resolve().stem
    mkdir_if_missing(result_root)

    dataloader = LoadVideo(
        opt.input_video,
        opt,
    )

    model = init_eval_model(opt)

    logger.info('Starting tracking...')

    result_filename = str(Path(result_root) / result_name / '.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else str(result_root / 'frame')
    try:
        eval_seq(
            opt,
            dataloader,
            'mot',
            result_filename,
            net=model,
            save_dir=frame_dir,
            frame_rate=frame_rate,
        )
    except TypeError as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = Path(result_root) / result_name / '.mp4'
        cmd_str = f"ffmpeg -f image2 -i {result_root / 'frame'}/%05d.jpg -c:v copy {output_video_path}"
        os.system(cmd_str)


if __name__ == '__main__':
    config = default_config
    track(config)
