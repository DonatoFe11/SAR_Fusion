import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchmetrics import (
    Accuracy,
    Metric,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
    ConfusionMatrix,
)
from torchmetrics.detection import MeanAveragePrecision

from sarfusion.utils.structures import DataDict
from sarfusion.utils.structures import WrapperModelOutput
from sarfusion.utils import TryExcept, threaded
from sarfusion.utils.general import box_iou, denormalize_boxes, scale_boxes, xywh2xyxy
from sarfusion.utils.utils import itemize_tensor


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    save_dir=".",
    names=(),
    eps=1e-16,
    prefix="",
):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(
            -px, -conf[i], recall[:, 0], left=0
        )  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(
            px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1"
        )
        plot_mc_curve(
            px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision"
        )
        plot_mc_curve(
            px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall"
        )

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class DetectionConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        import seaborn as sn

        array = self.matrix / (
            (self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_ylabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


class WIoU_Scale:
    """monotonous: {
        None: origin v1
        True: monotonic FM v2
        False: non-monotonic FM v3
    }
    momentum: The momentum of running mean"""

    iou_mean = 1.0
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train:
            cls.iou_mean = (
                1 - cls._momentum
            ) * cls.iou_mean + cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def bbox_iou(
    box1,
    box2,
    xywh=True,
    GIoU=False,
    DIoU=False,
    CIoU=False,
    MDPIoU=False,
    feat_h=640,
    feat_w=640,
    eps=1e-7,
):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        mpdiou_hw_pow = feat_h**2 + feat_w**2
        return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU
    return iou  # IoU


def bbox_ioa(box1, box2, eps=1e-7):
    """Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(nx4)
    box2:       np.array of shape(mx4)
    returns:    np.array of shape(nxm)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (
        np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(
        0
    )

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (
        wh1.prod(2) + wh2.prod(2) - inter + eps
    )  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(
    px,
    py,
    save_dir=Path("mc_curve.png"),
    names=(),
    xlabel="Confidence",
    ylabel="Metric",
):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


class MetricCollection(torch.nn.ModuleDict):
    def __init__(self, metrics):
        super().__init__()
        for k, v in metrics.items():
            self[k] = v

    def forward(self, *args, **kwargs):
        return {name: metric(*args, **kwargs) for name, metric in self.items()}

    def compute(self, *args, **kwargs):
        return {name: metric.compute() for name, metric in self.items()}

    def reset(self):
        for metric in self.values():
            metric.reset()

    def update(self, *args, **kwargs):
        for metric in self.values():
            metric.update(*args, **kwargs)


def build_metrics(params):
    metrics = {}
    for key, value in params.items():
        metric = globals()[key](**value)
        metrics[key] = metric
    metrics = MetricCollection(metrics)
    return metrics


def build_evaluator(params, task="classification", **kwargs):
    if params is None:
        return EmptyEvaluator()
    metrics = build_metrics(params.get("metrics", {}))
    if task == "detection":
        evaluator = DetectionEvaluator(metrics, **kwargs)
    else:
        evaluator = Evaluator(metrics)
    return evaluator


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


class Evaluator(Metric):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def update(self, batch_dict: DataDict, result_dict: WrapperModelOutput):
        y_true = batch_dict.labels
        y_pred = result_dict.logits
        return self.metrics.update(y_pred, y_true)

    def compute(self):
        return self.metrics.compute()

    def reset(self):
        return self.metrics.reset()


class EmptyEvaluator(Evaluator):
    def __init__(self):
        super().__init__({})

    def update(self, *args, **kwargs):
        return {}

    def compute(self):
        return {}

    def reset(self):
        return {}


# class DetectionEvaluator(Evaluator):
#     def __init__(self, metrics, id2class):
#         super().__init__(metrics)
#         nc = len(id2class)
#         self.confusion_matrix = DetectionConfusionMatrix(nc=nc)
#         self.nc = nc
#         self.id2class = id2class
#         self.stats = []

#     def update(self, batch_dict: DataDict, result_dict: WrapperModelOutput):
#         if "logits" not in result_dict:
#             return {}
#         preds = result_dict.logits
#         targets = batch_dict.target
#         dims = batch_dict.dims
#         images = batch_dict.images
#         device = "cuda"
#         iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
#         niou = iouv.numel()
#         # Metrics
#         for si, pred in enumerate(preds):
#             labels = targets[targets[:, 0] == si, 1:]
#             nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
#             shape = dims[si][0]
#             correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

#             if npr == 0:
#                 if nl:
#                     self.stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
#                 continue

#             # Predictions
#             predn = pred.clone()
#             # scale_boxes(images[si].shape[1:], predn[:, :4], shape, dims[si][1])  # native-space pred

#             # Evaluate
#             if nl:
#                 width, height = shape
#                 labels[:, 1:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
#                 tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
#                 # scale_boxes(images[si].shape[1:], tbox, shape, dims[si][1])  # native-space labels
#                 labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
#                 correct = process_batch(predn, labelsn, iouv)
#             self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
#         self.metrics.update(preds, targets)
#         return self.compute()

#     def compute(self):
#         # Compute metrics
#         stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
#         if len(stats) and stats[0].any():
#             tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=self.id2class)
#             ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#             mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#             metrics = {
#                 'mAP': map,
#                 'mAP50': map50,
#                 'P': mp,
#                 'R': mr,
#                 'F1': 2 * mp * mr / (mp + mr),
#                 'AP': ap,
#                 'AP50': ap50,
#                 'AP_class': ap_class
#             }
#         else:
#             metrics = {
#                 'mAP': 0,
#                 'mAP50': 0,
#                 'P': 0,
#                 'R': 0,
#                 'F1': 0,
#                 'AP': np.zeros(self.nc),
#                 'AP50': np.zeros(self.nc),
#                 'AP_class': np.zeros(self.nc)
#             }
#         return {**metrics, **self.metrics.compute()}


class DetectionEvaluator(Evaluator):
    def __init__(self, metrics, id2class):
        super().__init__(metrics)
        nc = len(id2class)
        self.metrics.add_module(
            "mAP",
            MeanAveragePrecision(
                box_format="xywh", class_metrics=True
            ),
        )
        self.nc = nc
        self.id2class = id2class
        self.stats = []
        
    def compute(self):
        metrics = super().compute()
        # Linearize mAP
        mAP = metrics.pop("mAP")
        metrics = {**metrics, **mAP}
        if isinstance(metrics, dict):
            return {k: itemize_tensor(v) for k, v in metrics.items()}
        return metrics

    def update(self, batch_dict: DataDict, result_dict: WrapperModelOutput):
        batch_metrics = []
        target = []
        for label in batch_dict.labels:
            target.append(
                {
                    "boxes": label["boxes"],
                    "labels": label["class_labels"],
                }
            )
        predictions = result_dict['predictions']
            
        batch_metrics.append({"preds": predictions, "target": target})
            
        all_preds = []
        all_targets = []
        for batch in batch_metrics:
            all_preds.extend(batch["preds"])
            all_targets.extend(batch["target"])
    
        self.metrics.update(preds=all_preds, target=all_targets)
