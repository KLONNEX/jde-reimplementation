# JDE

This is the clear PyTorch re-implementation of the JDE model from the
[original code](https://github.com/Zhongdao/Towards-Realtime-MOT) with some improvements.

<details>
<summary>Description</summary>

Paper with introduced JDE model is dedicated to the improving efficiency of an MOT system.
It's introduce an early attempt that Jointly learns the Detector and Embedding model (JDE) in a single-shot deep network.
In other words, the proposed JDE employs a single network to simultaneously output detection results and the corresponding appearance embeddings of the detected boxes.
In comparison, SDE methods and two-stage methods are characterized by re-sampled pixels (bounding boxes) and feature maps, respectively.
Both the bounding boxes and feature maps are fed into a separate re-ID model for appearance feature extraction.
Method is near real-time while being almost as accurate as the SDE methods.

</details>

<details>
<summary>Architecture</summary>

Architecture of the JDE is the Feature Pyramid Network (FPN).
FPN makes predictions from multiple scales, thus bringing improvement in pedestrian detection where the scale of targets varies a lot.
An input video frame first undergoes a forward pass through a backbone network to obtain feature maps at three scales, namely, scales with 1/32, 1/16 and 1/8 down-sampling rate, respectively.
Then, the feature map with the smallest size (also the semantically strongest features) is up-sampled and fused with the feature map from the second smallest scale by skip connection, and the same goes for the other scales.
Finally, prediction heads are added upon fused feature maps at all the three scales.
A prediction head consists of several stacked convolutional layers and outputs a dense prediction map of size (6A + D) × H × W, where A is the number of anchor templates assigned to this scale, and D is the dimension of the embedding.

</details>

<details>
<summary>Summary</summary>

| Parameters                 | GPU (1p)                                                                            |
| -------------------------- |-----------------------------------------------------------------------------------  |
| Model                      | JDE (1088*608)                                                                      |
| Hardware                   | 1 Nvidia RTX 2080 Ti, AMD Ryzen Threadripper 1950x 16-Core @ 3.40 GHz               |
| Dataset                    | Joint Dataset (see `DATASET_ZOO.md`)                                                |
| Training Parameters        | epoch=30, batch_size=4 (per device), lr=0.00125, momentum=0.9, weight_decay=0.0001  |
| Optimizer                  | SGD                                                                                 |
| Loss Function              | SmoothL1Loss, SoftmaxCrossEntropyWithLogits (and apply auto-balancing loss strategy)|
| Outputs                    | Tensor of bbox cords, conf, class, emb                                              |
| Speed                      | ~ 1.4 hours/epoch                                                                   |
| Total time                 | ~ 42 hours                                                                          |

</details>

## Dataset

Used a large-scale training set by putting together six publicly available datasets on pedestrian detection, MOT and person search.

Datasets preparations are described in [DATASET_ZOO.md](DATASET_ZOO.md).

Datasets size: 134G, 1 object category (pedestrian).

Note: `--dataset_root` is used as an entry point for all datasets, used for training and evaluating this model.

Organize your dataset structure as follows:

```text
.
└─dataset_root/
  ├─Caltech/
  ├─Cityscapes/
  ├─CUHKSYSU/
  ├─ETHZ/
  ├─MOT16/
  ├─MOT17/
  └─PRW/
```

## Training

You can follow the steps below for training and evaluation, in particular, before training,
you need to install `requirements.txt` by following command `pip install -r requirements.txt`.

All trainings will start from pre-trained backbone
([link](https://drive.google.com/file/d/1keZwVIfcWmxfTiswzOKUwkUz2xjvTvfm/view) for download).

```bash
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATASET_ROOT] [BACKBONE_PATH]
```

- DEVICE_ID - device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))
- BACKBONE_PATH - Path to the downloaded pre-trained darknet53 checkpoint.

The above command will run in the background, you can view the result through the generated standalone_train.log file.
After training, you can get the training loss and time logs in chosen logs_dir.

The model checkpoints will be saved in LOGS_CKPT_DIR directory.

## Evaluation

Tracking ability of the model is tested on the train part of the MOT16 dataset (doesn't use during training).

To start tracker evaluation run the command below.

```bash
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - device ID
- CKPT_URL - Path to the trained JDE model
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

> Note: the script expects that the DATASET_ROOT directory contains the MOT16 sub-folder.

The above python command will run in the background. The validation logs will be saved in "eval.log".

For more details about `motmetrics`, you can refer to [MOT benchmark](https://motchallenge.net/).

## Inference

To compile video from frames with predicted bounding boxes, you need to install `ffmpeg` by using
`sudo apt-get install ffmpeg`. Video compiling will happen automatically.

```bash
python infer.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --input_video [INPUT_VIDEO]
```

Results of the inference will be saved into default `./results` folder, logs will be shown at command line.

## Citations

[Paper](https://arxiv.org/pdf/1909.12605.pdf): Towards Real-Time Multi-Object Tracking. Department of Electronic Engineering, Tsinghua University
