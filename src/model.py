"""YOLOv3 based on DarkNet."""
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn import functional as F

from cfg.config import config as default_config
from src.darknet import DarkNet
from src.darknet import ResidualBlock
from src.modules import YoloBlock
from src.modules import _conv_bn_leaky
from src.utils import create_anchors_vec
from src.utils import decode_delta


class YOLOv3(nn.Module):
    """
    YOLOv3 Network.

    Args:
        backbone (nn.Module): Backbone Network.
        backbone_shape (list): Backbone output channels shape.
        out_channel (int): Output channel.

    Returns:
       small_feature (Tensor): Feature_map with shape (batch_size, backbone_shape[2], h/8, w/8).
       medium_feature (Tensor): Feature_map with shape (batch_size, backbone_shape[3], h/16, w/16).
       big_feature (Tensor): Feature_map with shape (batch_size, backbone_shape[4], h/32, w/32).
    """
    def __init__(self, backbone,  backbone_shape, out_channel):
        super().__init__()

        self.out_channel = out_channel
        self.backbone = backbone

        self.backblock0 = YoloBlock(
            in_channels=backbone_shape[-1],  # 1024
            out_chls=backbone_shape[-2],  # 512
            out_channels=out_channel,  # 24
        )

        self.conv1 = _conv_bn_leaky(
            in_channels=backbone_shape[-2],  # 1024
            out_channels=backbone_shape[-2] // 2,  # 512
            kernel_size=1,
        )
        self.backblock1 = YoloBlock(
            in_channels=backbone_shape[-2] + backbone_shape[-3],  # 768
            out_chls=backbone_shape[-3],  # 256
            out_channels=out_channel,  # 24
        )

        self.conv2 = _conv_bn_leaky(
            in_channels=backbone_shape[-3],  # 256
            out_channels=backbone_shape[-3] // 2,  # 128
            kernel_size=1,
        )
        self.backblock2 = YoloBlock(
            in_channels=backbone_shape[-3] + backbone_shape[-4],  # 384
            out_chls=backbone_shape[-4],  # 128
            out_channels=out_channel,  # 24
        )

        self.freeze_bn()

    def freeze_bn(self):
        """Freeze batch norms."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, inp):
        """
        Feed forward image to FPN to get 3 feature maps from different scales.

        Args:
            inp (Tensor): Input image (batch_size, 3, H, W).

        Returns:
            out (list): Feature maps (3) from different FPN levels.
        """
        img_hight, img_width = inp.shape[2:]

        feature_map_s, feature_map_m, feature_map_b = self.backbone(inp)
        con_s, output_s, emb_s = self.backblock0(feature_map_b)

        con_s = self.conv1(con_s)

        ups_s = F.interpolate(con_s, size=(img_hight // 16, img_width // 16), mode='nearest')
        con_s = torch.cat((ups_s, feature_map_m), dim=1)
        con_m, output_m, emb_m = self.backblock1(con_s)

        con_m = self.conv2(con_m)

        ups_m = F.interpolate(con_m, size=(img_hight // 8, img_width // 8), mode='nearest')
        con_b = torch.cat((ups_m, feature_map_s), dim=1)
        _, output_b, emb_b = self.backblock2(con_b)

        small_feature = torch.cat((output_s, emb_s), dim=1)
        medium_feature = torch.cat((output_m, emb_m), dim=1)
        big_feature = torch.cat((output_b, emb_b), dim=1)

        out = small_feature, medium_feature, big_feature

        return out


class YOLOLayer(nn.Module):
    """
    Head for loss calculation of classification confidence,
    bbox regression and ids embedding learning .

    Args:
        anchors (list): Absolute sizes of anchors (w, h).
        nid (int): Number of identities in whole train datasets.
        emb_dim (int): Size of embedding.
        nc (int): Number of ground truth classes.

    Returns:
        loss (Tensor): Auto balanced loss, calculated from conf, bbox and ids.
    """

    def __init__(
            self,
            anchors,
            nid,
            emb_dim,
            nc=default_config.num_classes,
    ):
        super().__init__()
        self.na = len(anchors)  # Number of anchors
        self.nc = nc  # Number of classes
        self.nid = nid  # Number of identities
        self.emb_dim = emb_dim  # Embedding size

        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.softmax_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # Set trainable parameters for loss computation
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nid - 1)

    def forward(self, p_cat, tconf, tbox, tids, classifier):
        """
        Feed forward output from the FPN,
        calculate confidence loss, bbox regression loss, target id loss,
        apply auto-balancing loss strategy.
        """
        # Get detections and embeddings from model concatenated output.
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nb, ngh, ngw = p.shape[0], p.shape[-2], p.shape[-1]

        p = p.view(nb, self.na, self.nc + 5, ngh, ngw).permute(0, 1, 3, 4, 2)  # prediction
        p_emb = p_emb.permute(0, 2, 3, 1)
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)

        mask = tconf > 0

        # Compute losses
        nm = mask.sum().int()  # number of anchors (assigned to targets)

        if nm > 0:
            lbox = self.smooth_l1_loss(p_box[mask], tbox[mask])
        else:
            lbox, lconf = torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0])

        lconf = self.softmax_loss(p_conf, tconf)

        emb_mask = mask.max(1)[0]

        tids = tids.max(1)[0]
        tids = tids[emb_mask].squeeze()
        embedding = p_emb[emb_mask]
        embedding = self.emb_scale * F.normalize(embedding)

        if len(embedding) > 1:
            logits = classifier(embedding)
            lid = self.id_loss(logits, tids)
            if not lid >= 0:
                lid = torch.cuda.FloatTensor([0])
        else:
            lid = torch.cuda.FloatTensor([0])

        # Apply auto-balancing loss strategy
        loss = torch.exp((-1) * self.s_r) * lbox + \
               torch.exp((-1) * self.s_c) * lconf + \
               torch.exp((-1) * self.s_id) * lid + \
               (self.s_r + self.s_c + self.s_id)

        loss = loss.squeeze() * 0.5

        return loss


class JDE(nn.Module):
    """
    JDE Network.

    Args:
        extractor (nn.Module): Backbone, which extracts feature maps.
        config (class): Config with model and training params.
        nid (int): Number of identities in whole train datasets.
        ne (int): Size of embedding.
    """

    def __init__(self, extractor, config, nid, ne):
        super().__init__()
        anchors = config.anchor_scales
        anchors1 = anchors[0:4]
        anchors2 = anchors[4:8]
        anchors3 = anchors[8:12]

        self.backbone = extractor

        # Set loss cell layers for different scales
        self.head_s = YOLOLayer(anchors3, nid, ne)
        self.head_m = YOLOLayer(anchors2, nid, ne)
        self.head_b = YOLOLayer(anchors1, nid, ne)

        # Set classifier for embeddings
        self.classifier = nn.Linear(in_features=ne, out_features=nid)

    def forward(
            self,
            images,
            tconf_s,
            tbox_s,
            tid_s,
            tconf_m,
            tbox_m,
            tid_m,
            tconf_b,
            tbox_b,
            tid_b,
    ):
        """
        Feed forward image to FPN, get 3 feature maps with different sizes,
        put it into 3 heads, corresponding to size, get auto-balanced losses.

        Args:
            images (Tensor): Input batch of images (bs, 3, h, w).
            tconf_s (Tensor): Small confidence grid (bs, na, h // 32, w // 32).
            tbox_s (Tensor): Small bounding box grid (bs, na, h // 32, w // 32, 4).
            tid_s (Tensor): Small id grid (bs, na, h // 32, w // 32).
            tconf_m (Tensor): Medium confidence grid (bs, na, h // 16, w // 16).
            tbox_m (Tensor): Medium bounding box grid (bs, na, h // 16, w // 16, 4).
            tid_m (Tensor): Medium id grid (bs, na, h // 16, w // 16).
            tconf_b (Tensor): Big confidence grid (bs, na, h // 8, w // 8).
            tbox_b (Tensor): Big bounding box grid (bs, na, h // 8, w // 8, 4).
            tid_b (Tensor): Big id grid (bs, na, h // 8, w // 8).

        Returns:
            loss (Tensor): Mean loss from 3 heads.
        """
        # Apply FPN to image to get 3 feature map with different scales
        small, medium, big = self.backbone(images)

        # Calculate losses for each feature map
        out_s = self.head_s(small, tconf_s, tbox_s, tid_s, self.classifier)
        out_m = self.head_m(medium, tconf_m, tbox_m, tid_m, self.classifier)
        out_b = self.head_b(big, tconf_b, tbox_b, tid_b, self.classifier)

        loss = (out_s + out_m + out_b) / 3

        return loss


class YOLOLayerEval(nn.Module):
    """
    Head for detection and tracking.

    Args:
        anchor (list): Absolute sizes of anchors (w, h).
        nc (int): Number of ground truth classes.

    Returns:
        prediction: Model predictions for confidences, boxes and embeddings.
    """

    def __init__(
            self,
            anchor,
            stride,
            nc=default_config.num_classes,
    ):
        super().__init__()
        self.na = len(anchor)  # number of anchors (4)
        self.nc = nc  # number of classes (1)
        self.anchor_vec = torch.FloatTensor(anchor).cuda()
        self.stride = stride

    def forward(self, p_cat):
        """
        Feed forward output from the FPN,
        calculate prediction corresponding to anchor.
        """
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nb, ngh, ngw = p.shape[0], p.shape[-2], p.shape[-1]

        p = p.view(nb, self.na, self.nc + 5, ngh, ngw).permute(0, 1, 3, 4, 2)  # prediction
        p_emb = p_emb.permute(0, 2, 3, 1)
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # conf
        p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
        p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1, self.na, 1, 1, 1), dim=-1)

        p_cls = torch.zeros((nb, self.na, ngh, ngw, 1)).cuda()  # temp
        p = torch.cat((p_box, p_conf, p_cls, p_emb), dim=-1)

        # Decode bbox delta to the absolute cords
        p_1 = decode_delta(p[..., :4], self.anchor_vec)
        p_1 = p_1 * self.stride

        p = torch.cat((p_1, p[..., 4:]), dim=-1)
        prediction = p.reshape(nb, -1, p.shape[-1])

        return prediction


class JDEeval(nn.Module):
    """
     JDE Network.

     Note:
         backbone = YOLOv3 with darknet53.
         head = 3 similar heads for each feature map size.

     Returns:
         output: Tensor with concatenated outputs from each head.
         output_top_k: Output tensor of top_k best proposals by confidence.
    """

    def __init__(self, extractor, config):
        super().__init__()
        anchors, strides = create_anchors_vec(config.anchor_scales)

        self.backbone = extractor

        self.head_s = YOLOLayerEval(anchors[0], strides[0])
        self.head_m = YOLOLayerEval(anchors[1], strides[1])
        self.head_b = YOLOLayerEval(anchors[2], strides[2])

        # Top K to cut extra predictions to save time
        # of .cpu() transfer from .cuda() device
        self.k = 800

    def forward(self, images):
        """
        Feed forward image to FPN, get 3 feature maps with different sizes,
        put them into 3 heads, corresponding to size,
        get concatenated output of proposals.
        """
        small, medium, big = self.backbone(images)

        out_s = self.head_s(small)
        out_m = self.head_m(medium)
        out_b = self.head_b(big)

        output = torch.cat((out_s, out_m, out_b), dim=1)

        _, top_k_indices = torch.topk(output[:, :, 4], self.k, sorted=False)
        output_top_k = output[0][top_k_indices]

        return output, output_top_k


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

    load_darknet_weights(model=backbone, weights=cfg.pretrained_path)

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


def init_eval_model(cfg):
    """
    Initialize model, and load weights into backbone.

    Args:
        cfg: Config parameters.

    Returns:
        network: Compiled train model with loss.
    """
    backbone = DarkNet(
        block=ResidualBlock,
        layer_nums=cfg.backbone_layers,
        in_channels=cfg.backbone_input_shape,
        out_channels=cfg.backbone_output_shape,
    )

    net = YOLOv3(
        backbone=backbone,
        backbone_shape=cfg.backbone_output_shape,
        out_channel=cfg.out_channel,
    )

    network = JDEeval(
        extractor=net,
        config=cfg,
    )

    weights = torch.load(cfg.ckpt_url)['model']
    weights_keys = list(weights.keys())[-11:]
    for key in weights_keys:
        weights.pop(key)
    network.load_state_dict(weights)
    network.cuda().eval()

    return network


def load_darknet_weights(model, weights):
    """
    Load backbone weights into model from existing file
    or download and load if not exist.

    Args:
        model: Inited train model with backbone.
        weights (str): Path to backbone weights file.
    """
    # Parses and loads the weights stored in 'weights'
    Path(weights).resolve().parent.mkdir(parents=True, exist_ok=True)
    weights_file = Path(weights).name

    # Try to download weights if not available locally
    if not Path(weights).exists():
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Open the weights file
    with Path(weights).open('rb') as fp:
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        model.header_info = header

        model.seen = header[3]  # number of images seen during training
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights

    ptr = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Load conv. weights
            num_w = module.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(module.weight)
            module.weight.data.copy_(conv_w)
            ptr += num_w

        elif isinstance(module, nn.BatchNorm2d):
            # Load BN bias, weights, running mean and running variance
            num_b = module.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(module.bias)
            module.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(module.weight)
            module.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(module.running_mean)
            module.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(module.running_var)
            module.running_var.data.copy_(bn_rv)
            ptr += num_b

    print("Loading of backbone weights succeed.")
