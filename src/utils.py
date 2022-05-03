import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def xyxy2xywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h],
    where x, y are coordinates of center, (x1, y1) and (x2, y2)
    are coordinates of bottom left and top right respectively.
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2],
    where x, y are coordinates of center, (x1, y1) and (x2, y2)
    are coordinates of bottom left and top right respectively.
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # Bottom left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # Top right y
    return y


def scale_coords(img_size, coords, img0_shape):
    """
    Rescale x1, y1, x2, y2 to image size.
    """
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    cords_max = np.max(coords[:, :4])
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=cords_max)
    return coords


def build_targets_thres(target, anchor_wh, na, ngh, ngw):
    """
    Build grid of targets confidence mask, bbox delta and id with thresholds.

    Args:
        target (np_array): Targets bbox cords and ids.
        anchor_wh (np_array): Resized anchors for map size.
        na (int): Number of anchors.
        ngh (int): Map height.
        ngw (int): Map width.

    Returns:
        tconf (np_array): Mask with bg (0), gt (1) and ign (-1) indices. Shape (na, ngh, ngw).
        tbox (np_array): Targets delta bbox values. Shape (na, ngh, ngw, 4).
        tid (np_array): Grid with id for every cell. Shape (na, ngh, ngw).
    """
    id_thresh = 0.5
    fg_thresh = 0.5
    bg_thresh = 0.4

    bg_id = -1  # Background id

    tbox = np.zeros((na, ngh, ngw, 4), dtype=np.float32)  # Fill grid with zeros bbox cords
    tconf = np.zeros((na, ngh, ngw), dtype=np.int32)  # Fill grid with zeros confidence
    tid = np.full((na, ngh, ngw), bg_id, dtype=np.int32)  # Fill grid with background id

    t = target
    t_id = t[:, 1].copy().astype(np.int32)
    t = t[:, [0, 2, 3, 4, 5]]

    # Convert relative cords for map size
    gxy, gwh = t[:, 1:3].copy(), t[:, 3:5].copy()
    gxy[:, 0] = gxy[:, 0] * ngw
    gxy[:, 1] = gxy[:, 1] * ngh
    gwh[:, 0] = gwh[:, 0] * ngw
    gwh[:, 1] = gwh[:, 1] * ngh
    gxy[:, 0] = np.clip(gxy[:, 0], a_min=0, a_max=ngw - 1)
    gxy[:, 1] = np.clip(gxy[:, 1], a_min=0, a_max=ngh - 1)

    gt_boxes = np.concatenate((gxy, gwh), axis=1)  # Shape (num of targets, 4), 4 is (xc, yc, w, h)

    # Apply anchor to each cell of the grid
    anchor_mesh = generate_anchor(ngh, ngw, anchor_wh)  # Shape (na, 4, ngh, ngw)
    anchor_list = anchor_mesh.transpose(0, 2, 3, 1).reshape(-1, 4)  # Shape (na x ngh x ngw, 4)

    # Compute anchor iou with ground truths bboxes
    iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (na x ngh x ngw, Ng)
    max_gt_index = iou_pdist.argmax(axis=1)   # Shape (na x ngh x ngw)
    iou_max = iou_pdist.max(axis=1)   # Shape (na x ngh x ngw)

    iou_map = iou_max.reshape(na, ngh, ngw)
    gt_index_map = max_gt_index.reshape(na, ngh, ngw)

    # Fill tconf by thresholds
    id_index = iou_map > id_thresh
    fg_index = iou_map > fg_thresh
    bg_index = iou_map < bg_thresh
    ign_index = (iou_map < fg_thresh) * (iou_map > bg_thresh)  # Search unclear cells
    tconf[fg_index] = 1
    tconf[bg_index] = 0
    tconf[ign_index] = -1  # Index to ignore unclear cells

    # Take ground truths with mask
    gt_index = gt_index_map[fg_index]
    gt_box_list = gt_boxes[gt_index]
    gt_id_list = t_id[gt_index_map[id_index]]
    if np.sum(fg_index) > 0:
        tid[id_index] = gt_id_list
        fg_anchor_list = anchor_list.reshape((na, ngh, ngw, 4))[fg_index]
        delta_target = encode_delta(gt_box_list, fg_anchor_list)
        tbox[fg_index] = delta_target

    return tconf, tbox, tid


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes.
    """
    n, m = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(np.expand_dims(b1_x1, 1), b2_x1)
    inter_rect_y1 = np.maximum(np.expand_dims(b1_y1, 1), b2_y1)
    inter_rect_x2 = np.minimum(np.expand_dims(b1_x2, 1), b2_x2)
    inter_rect_y2 = np.minimum(np.expand_dims(b1_y2, 1), b2_y2)

    # Intersection area
    i_r_x = inter_rect_x2 - inter_rect_x1
    i_r_y = inter_rect_y2 - inter_rect_y1
    inter_area = np.clip(i_r_x, 0, np.max(i_r_x)) * np.clip(i_r_y, 0, np.max(i_r_y))

    # Union Area
    b1_area = np.broadcast_to(((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1, 1), (n, m))
    b2_area = np.broadcast_to(((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1, -1), (n, m))

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def generate_anchor(ngh, ngw, anchor_wh):
    """
    Generate anchor for every cell in grid.
    """
    na = len(anchor_wh)
    yy, xx = np.meshgrid(np.arange(ngh), np.arange(ngw), indexing='ij')

    mesh = np.stack([xx, yy], axis=0)  # Shape 2, ngh, ngw
    mesh = np.tile(np.expand_dims(mesh, 0), (na, 1, 1, 1)).astype(np.float32)  # Shape na, 2, ngh, ngw
    anchor_offset_mesh = np.tile(np.expand_dims(np.expand_dims(anchor_wh, -1), -1), (1, 1, ngh, ngw))  # Shape na, 2, ngh, ngw
    anchor_mesh = np.concatenate((mesh, anchor_offset_mesh), axis=1)  # Shape na, 4, ngh, ngw
    return anchor_mesh


def encode_delta(gt_box_list, fg_anchor_list):
    """
    Calculate delta for bbox center, width, height.
    """
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:, 1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:, 3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)

    return np.stack([dx, dy, dw, dh], axis=1)


def create_grids(anchors, img_size, ngw):
    """
    Resize anchor according to image size and feature map size.

    Note:
        Ratio of feature maps dimensions if 1:3 such as anchors.
        Thus, it's enough to calculate stride per one dimension.
    """
    stride = img_size[0] / ngw
    anchor_vec = np.array(anchors) / stride

    return anchor_vec, stride


def build_thresholds(
        labels,
        anchor_vec_s,
        anchor_vec_m,
        anchor_vec_b,
        img_size=(1088, 608),
):
    """
    Build thresholds for all feature map sizes.
    """
    h, w = img_size[1], img_size[0]

    s = build_targets_thres(labels, anchor_vec_s, 4, h // 32, w // 32)
    m = build_targets_thres(labels, anchor_vec_m, 4, h // 16, w // 16)
    b = build_targets_thres(labels, anchor_vec_b, 4, h // 8, w // 8)

    return s, m, b


def create_anchors_vec(anchors, img_size=(1088, 608)):
    """
    Create anchor vectors for every feature map size.
    """
    anchors1 = anchors[0:4]
    anchors2 = anchors[4:8]
    anchors3 = anchors[8:12]
    anchor_vec_s, stride_s = create_grids(anchors3, img_size, 34)
    anchor_vec_m, stride_m = create_grids(anchors2, img_size, 68)
    anchor_vec_b, stride_b = create_grids(anchors1, img_size, 136)

    anchors = (anchor_vec_s, anchor_vec_m, anchor_vec_b)
    strides = (stride_s, stride_m, stride_b)

    return anchors, strides


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.

    Args:
        prediction (np.array): All predictions from model output.
        conf_thres (float): Threshold for confidence.
        nms_thres (float): Threshold for iou into nms.

    Returns:
        output (np.array): Predictions with shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = np.squeeze(v.nonzero())
        if v.ndim == 0:
            v = np.expand_dims(v, 0)

        pred = pred[v]

        # If none are remaining => process next image
        npred = pred.shape[0]
        if not npred:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        bboxes = np.concatenate((pred[:, :4], np.expand_dims(pred[:, 4], -1)), axis=1)
        nms_indices = nms(bboxes, nms_thres)
        det_max = pred[nms_indices]

        if det_max.size > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else np.concatenate((output[image_i], det_max))

    return output


def nms(dets, thresh):
    """
    Non-maximum suppression with threshold.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Computes the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        tp (list): True positives.
        conf (list): Objectness value from 0-1.
        pred_cls (np.array): Predicted object classes.
        target_cls (np.array): True object classes.

    Returns:
        ap (np.array): The average precision as computed in py-faster-rcnn.
        unique classes (np.array): Classes of predictions.
        r (np.array): Recall.
        p (np.array): Precision.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue

        if (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """
    Computes the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (np.array): The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def load_darknet_weights(model, weights):
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


def collate_fn(raw_batch):
    """
    Convert batch to the input format.

    Args:
        raw_batch (list): Batch with cpu tensors.

    Returns:
        prepared_batch (list): Batch with cuda and converted to dtype tensors.
    """
    imgs, tconf_s, tbox_s, tid_s, tconf_m, tbox_m, tid_m, tconf_b, tbox_b, tid_b = raw_batch
    prepared_batch = (
        imgs.cuda().float(),
        tconf_s.cuda().long(),
        tbox_s.cuda().float(),
        tid_s.cuda().long(),
        tconf_m.cuda().long(),
        tbox_m.cuda().float(),
        tid_m.cuda().long(),
        tconf_b.cuda().long(),
        tbox_b.cuda().float(),
        tid_b.cuda().long(),
    )

    return prepared_batch
