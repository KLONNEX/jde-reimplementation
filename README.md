# JDE re-implementation

- [Original](#original)
- [JDE Description](#jde-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Training Process](#training-process)
        - [Standalone Training](#standalone-training)
        - [Distribute Training](#distribute-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)

# [Original](#contents)

This is the significantly clear PyTorch re-implementation of the JDE model from the
[original code](https://github.com/Zhongdao/Towards-Realtime-MOT) with some improvements.

[Paper](https://arxiv.org/pdf/1909.12605.pdf): Towards Real-Time Multi-Object Tracking. Department of Electronic Engineering, Tsinghua University

# [JDE Description](#contents)

Paper with introduced JDE model is dedicated to the improving efficiency of an MOT system.
It's introduce an early attempt that Jointly learns the Detector and Embedding model (JDE) in a single-shot deep network.
In other words, the proposed JDE employs a single network to simultaneously output detection results and the corresponding appearance embeddings of the detected boxes.
In comparison, SDE methods and two-stage methods are characterized by re-sampled pixels (bounding boxes) and feature maps, respectively.
Both the bounding boxes and feature maps are fed into a separate re-ID model for appearance feature extraction.
Method is near real-time while being almost as accurate as the SDE methods.

# [Model Architecture](#contents)

Architecture of the JDE is the Feature Pyramid Network (FPN).
FPN makes predictions from multiple scales, thus bringing improvement in pedestrian detection where the scale of targets varies a lot.
An input video frame first undergoes a forward pass through a backbone network to obtain feature maps at three scales, namely, scales with 1/32, 1/16 and 1/8 down-sampling rate, respectively.
Then, the feature map with the smallest size (also the semantically strongest features) is up-sampled and fused with the feature map from the second smallest scale by skip connection, and the same goes for the other scales.
Finally, prediction heads are added upon fused feature maps at all the three scales.
A prediction head consists of several stacked convolutional layers and outputs a dense prediction map of size (6A + D) × H × W, where A is the number of anchor templates assigned to this scale, and D is the dimension of the embedding.

# [Dataset](#contents)

Used a large-scale training set by putting together six publicly available datasets on pedestrian detection, MOT and person search.

These datasets can be categorized into two types: ones that only contain bounding box annotations, and ones that have both bounding box and identity annotations.
The first category includes the ETH dataset and the CityPersons (CP) dataset. The second category includes the CalTech (CT) dataset, MOT16 (M16) dataset, CUHK-SYSU (CS) dataset and PRW dataset.
Training subsets of all these datasets are gathered to form the joint training set, and videos in the ETH dataset that overlap with the MOT-16 test set are excluded for fair evaluation.

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

Information about train part of dataset.

| Dataset | ETH |  CP |  CT | M16 |  CS | PRW | Total |
| :------:|:---:|:---:|:---:|:---:|:---:|:---:|:-----:|
| # img   |2K   |3K   |27K  |53K  |11K  |6K   |54K    |
| # box   |17K  |21K  |46K  |112K |55K  |18K  |270K   |
| # ID    |-    |-    |0.6K |0.5K |7K   |0.5K |8.7K   |

# [Quick Start](#contents)

You can follow the steps below for training and evaluation, in particular, before training,
you need to install `requirements.txt` by following command `pip install -r requirements.txt`.

> If an error occurred, update pip by `pip install --upgrade pip` and try again.
> If it didn't help install packages manually by using `pip install {package from requirements.txt}`.

All trainings will starts from pre-trained backbone, which will automatically downloaded by running training scripts.

```bash
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATASET_ROOT]
```

- DEVICE_ID - device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

## [Training Process](#contents)

### Standalone Training

Note: For all trainings necessary to use pretrained backbone darknet53.

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATASET_ROOT]
```

- DEVICE_ID - device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

The above command will run in the background, you can view the result through the generated standalone_train.log file.
After training, you can get the training loss and time logs in chosen logs_dir.

The model checkpoints will be saved in LOGS_CKPT_DIR directory.

## [Evaluation Process](#contents)

### Evaluation

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

```text
None
```

To evaluate detection ability (get mAP, Precision and Recall metrics) of the model, run command below.

```bash
python eval_detect.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --dataset_root [DATASET_ROOT]
```

Results of evaluation will be visualized at command line.

## [Inference Process](#contents)

### Usage

To compile video from frames with predicted bounding boxes, you need to install `ffmpeg` by using
`sudo apt-get install ffmpeg`. Video compiling will happen automatically.

```bash
python infer.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --input_video [INPUT_VIDEO]
```

### Result

Results of the inference will be saved into default `./results` folder, logs will be shown at command line.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

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

### Evaluation Performance

| Parameters          | GPU (1p)                                                              |
| ------------------- |-----------------------------------------------------------------------|
| Model               | JDE (1088*608)                                                        |
| Resource            | 1 Nvidia RTX 2080 Ti, AMD Ryzen Threadripper 1950x 16-Core @ 3.40 GHz |
| Dataset             | MOT-16 (train)                                                        |
| Batch_size          | 1                                                                     |
| Outputs             | Metrics, .txt predictions                                             |
| FPS                 |                                                                       |
| Metrics             |                                                                       |
