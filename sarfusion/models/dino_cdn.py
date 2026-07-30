"""
dino_cdn.py
-----------
Contrastive DeNoising (CDN) query generator for DINO-Fusion.

This module is self-contained: it only needs torch and produces tensors
whose shapes are dictated by the DeformableDetr decoder contract.

DeformableDetr decoder contract (two-stage DINO):
  inputs_embeds                  : (B, Q,   d_model)   ← content ("target")
  object_queries_position_embeds : (B, Q,   d_model)   ← positional ("query_embed")
  reference_points               : (B, Q,   2 or 4)    ← normalised cx,cy[,w,h]
  decoder_attention_mask         : (Q_total, Q_total)   ← float additive mask or None

We prepend N_dn CDN slots before the N_match matching slots, giving
Q_total = N_dn + N_match.

At inference time nothing is called; the matching slots behave exactly as
they do in the base Deformable DETR.

Reference: DINO paper §3.1 and the official IDEA-Research implementation
  https://github.com/IDEACVR/DINO/blob/main/models/dino/dn_components.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public dataclass returned by build_cdn_queries()
# ---------------------------------------------------------------------------

@dataclass
class CDNTargets:
    """
    Everything the forward() override and the loss need.

    Shapes
    ------
    dn_content    : (B, N_dn, d_model)  – content embeddings for DN slots
    dn_ref_points : (B, N_dn, 4)        – noised reference boxes (cx,cy,w,h) in [0,1]
    dn_target_boxes: (B, N_dn, 4)       – original boxes used as CDN reconstruction targets
    attn_mask     : (N_total, N_total)  – additive float mask (0 = attend, -inf = block)
    num_dn        : int                 – N_dn = 2 * num_dn_groups * max_gt_per_image
    num_groups    : int                 – num_dn_groups (used by loss)
    max_gt        : int                 – max GT objects per image in this batch
    gt_indices    : List[Tensor]        – per-image mapping: dn_slot_i → gt_box_j
    pos_neg_flag  : Tensor (B, N_dn)    – 1=positive DN slot, 0=negative DN slot
    """
    dn_content: torch.Tensor
    dn_ref_points: torch.Tensor
    dn_target_boxes: torch.Tensor
    dn_labels: torch.Tensor
    dn_target_labels: torch.Tensor
    valid_slot_mask: torch.Tensor
    attn_mask: torch.Tensor
    num_dn: int
    num_groups: int
    max_gt: int
    gt_indices: List[torch.Tensor]
    pos_neg_flag: torch.Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse of sigmoid, matching the HuggingFace version."""
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def _box_noise(
    boxes: torch.Tensor,
    box_noise_scale: float,
    device: torch.device,
    negative: bool = False,
) -> torch.Tensor:
    """
    Apply DINO's contrastive box perturbation in coordinate space.

    boxes : (N, 4)  cx,cy,w,h in [0,1]
    Returns noised boxes in [0,1], same shape.
    """
    # The official implementation perturbs xyxy corners by half the original
    # width/height. Positive magnitudes are in [0, 1), negative magnitudes in
    # [1, 2), which makes the two copies genuinely contrastive.
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )
    radius = torch.stack([w / 2, h / 2, w / 2, h / 2], dim=-1)
    signs = torch.randint(0, 2, xyxy.shape, device=device, dtype=torch.long)
    signs = signs.to(xyxy.dtype).mul_(2).sub_(1)
    magnitude = torch.rand_like(xyxy)
    if negative:
        magnitude = magnitude + 1.0
    xyxy = (xyxy + signs * magnitude * radius * box_noise_scale).clamp(0.0, 1.0)
    x0, y0, x1, y1 = xyxy.unbind(-1)
    return torch.stack(
        [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1
    )


def _label_noise(
    class_labels: torch.Tensor,
    num_classes: int,
    label_noise_prob: float,
) -> torch.Tensor:
    """
    Apply DINO's configured label-noise ratio (effective flip threshold:
    ``label_noise_prob * 0.5``).

    class_labels : (N,)  int64
    Returns noised labels, same shape.
    """
    if label_noise_prob == 0.0:
        return class_labels.clone()
    # DINO applies the configured label-noise ratio with the 0.5 factor used
    # by its official query preparation code.
    mask = (
        torch.rand(class_labels.shape, device=class_labels.device)
        < label_noise_prob * 0.5
    )
    random_labels = torch.randint_like(class_labels, 0, num_classes)
    return torch.where(mask, random_labels, class_labels)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_cdn_queries(
    targets: List[dict],
    # Model embedding modules – passed in, not owned here
    label_embeddings: torch.nn.Embedding,   # (num_classes+1, d_model) – the class embed table
    # CDN hyperparameters
    num_dn_groups: int = 5,
    label_noise_prob: float = 0.5,
    box_noise_scale: float = 1.0,
    # Model geometry (read from config, passed explicitly for clarity)
    num_queries: int = 300,
    d_model: int = 256,
    num_classes: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[CDNTargets]:
    """
    Build CDN query tensors for one training batch.

    Parameters
    ----------
    targets : list of dicts, one per image in the batch.
        Each dict must have:
            "class_labels" : LongTensor  (n_gt,)
            "boxes"        : FloatTensor (n_gt, 4)  cx,cy,w,h in [0,1]
    label_embeddings :
        The model's class embedding table.  We look up the noised class index
        to produce the content vector.  Shape (num_classes, d_model).
    num_dn_groups : int
        Number of CDN groups.  Each group contains one positive + one negative copy of
        every GT box.  Total DN slots = 2 * num_dn_groups * max_gt_per_image.
    label_noise_prob : float  ∈ [0,1]
        DINO label-noise ratio; the official preparation samples flips with
        threshold ``label_noise_prob * 0.5``.
    box_noise_scale : float
        Scale of the corner perturbation. Positive copies use magnitudes in [0, 1),
        contrastive-negative copies use [1, 2).
    num_queries : int   N_match  (matching queries, unchanged)
    d_model     : int   Hidden dimension.
    num_classes : int   Number of foreground classes (without background/no-obj).
    device      : torch.device

    Returns
    -------
    CDNTargets  or  None if every image in the batch has zero GT objects.
    """
    batch_size = len(targets)
    if device is None:
        device = label_embeddings.weight.device

    # ------------------------------------------------------------------ #
    #  1. Gather per-image GT, determine max_gt for padding               #
    # ------------------------------------------------------------------ #
    gt_boxes_list:  List[torch.Tensor] = []   # each (n_i, 4)
    gt_labels_list: List[torch.Tensor] = []   # each (n_i,)

    for t in targets:
        boxes  = t["boxes"].to(device)   # (n_i, 4)
        labels = t["class_labels"].to(device).long()  # (n_i,)
        gt_boxes_list.append(boxes)
        gt_labels_list.append(labels)

    max_gt = max(b.shape[0] for b in gt_boxes_list)

    if max_gt == 0:
        # No GT in this batch – CDN is a no-op
        return None

    # N_dn = 2 groups (pos+neg) × num_dn_groups × max_gt
    num_dn = 2 * num_dn_groups * max_gt

    # ------------------------------------------------------------------ #
    #  2. Build padded GT tensors  (B, max_gt, 4/1)                      #
    # ------------------------------------------------------------------ #
    # boxes_pad  : (B, max_gt, 4)   padded with 0.5 (centre of image)
    # labels_pad : (B, max_gt)      padded with 0   (arbitrary, masked out)
    # valid_mask : (B, max_gt)      True where real GT exists
    boxes_pad = torch.full((batch_size, max_gt, 4), 0.5, device=device)
    labels_pad = torch.zeros(batch_size, max_gt, dtype=torch.long, device=device)
    valid_mask = torch.zeros(batch_size, max_gt, dtype=torch.bool, device=device)

    for i, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        n = boxes.shape[0]
        if n > 0:
            boxes_pad[i, :n]  = boxes
            labels_pad[i, :n] = labels
            valid_mask[i, :n] = True

    # ------------------------------------------------------------------ #
    #  3. Tile for num_dn_groups  →  (B, 2*num_dn_groups*max_gt, ...)   #
    # ------------------------------------------------------------------ #
    # Tile: repeat along the gt axis for each group, then interleave pos/neg.
    # Layout per group g  (0-indexed):
    #   slots [2g*max_gt  .. (2g+1)*max_gt - 1]  → positive copies
    #   slots [(2g+1)*max_gt .. (2g+2)*max_gt-1] → negative copies

    # boxes_tiled  : (B, num_dn_groups, 2, max_gt, 4)  → collapse last 3 dims later
    target_boxes_tiled = boxes_pad.unsqueeze(1).unsqueeze(2).expand(
        batch_size, num_dn_groups, 2, max_gt, 4
    ).clone()
    boxes_tiled = target_boxes_tiled.clone()
    labels_tiled = labels_pad.unsqueeze(1).unsqueeze(2).expand(
        batch_size, num_dn_groups, 2, max_gt
    ).clone()
    target_labels_tiled = labels_tiled.clone()
    valid_slots_tiled = valid_mask.unsqueeze(1).unsqueeze(2).expand(
        batch_size, num_dn_groups, 2, max_gt
    )

    # ------------------------------------------------------------------ #
    #  4. Add noise                                                       #
    # ------------------------------------------------------------------ #
    # Positive  (dim-index 0): small noise = box_noise_scale / 2
    # Negative  (dim-index 1): large noise = box_noise_scale

    for b in range(batch_size):
        n = gt_boxes_list[b].shape[0]
        if n == 0:
            continue
        for g in range(num_dn_groups):
            # --- positive ---
            pos_boxes = boxes_tiled[b, g, 0, :n]       # (n, 4)
            noised_pos = _box_noise(pos_boxes, box_noise_scale, device, negative=False)
            boxes_tiled[b, g, 0, :n] = noised_pos

            noised_labels = _label_noise(labels_tiled[b, g, 0, :n], num_classes, label_noise_prob)
            labels_tiled[b, g, 0, :n] = noised_labels

            # --- negative ---
            neg_boxes = boxes_tiled[b, g, 1, :n]       # (n, 4)
            noised_neg = _box_noise(neg_boxes, box_noise_scale, device, negative=True)
            boxes_tiled[b, g, 1, :n] = noised_neg
            noised_labels = _label_noise(labels_tiled[b, g, 1, :n], num_classes, label_noise_prob)
            labels_tiled[b, g, 1, :n] = noised_labels

    # Reshape to (B, num_dn, 4) and (B, num_dn)
    # Current shape: (B, num_dn_groups, 2, max_gt, ...)
    # → (B, num_dn_groups * 2 * max_gt, ...)
    dn_ref_points = boxes_tiled.reshape(batch_size, num_dn, 4).clamp(0.0, 1.0)
    dn_target_boxes = target_boxes_tiled.reshape(batch_size, num_dn, 4)
    dn_labels     = labels_tiled.reshape(batch_size, num_dn)       # (B, num_dn)
    dn_target_labels = target_labels_tiled.reshape(batch_size, num_dn)
    valid_slot_mask = valid_slots_tiled.reshape(batch_size, num_dn)

    # pos_neg_flag: 1 for positive slots, 0 for negative slots
    # Mark only real GT (per-image) as positive; padded slots stay negative.
    # Positive slots occupy even groups in the layout above:
    #   indices [g*2*max_gt .. g*2*max_gt + n_i - 1]
    pos_neg_flag = torch.zeros(batch_size, num_dn, device=device)
    for b in range(batch_size):
        n_i = int(valid_mask[b].sum().item())
        if n_i == 0:
            continue
        for g in range(num_dn_groups):
            start = g * 2 * max_gt
            end = start + n_i
            pos_neg_flag[b, start:end] = 1.0

    # ------------------------------------------------------------------ #
    #  5. Build content embeddings  (B, N_dn, d_model)                   #
    # ------------------------------------------------------------------ #
    embed_indices = dn_labels.clamp(0, num_classes - 1)   # (B, num_dn)
    dn_content = label_embeddings(embed_indices)           # (B, num_dn, d_model)

    # ------------------------------------------------------------------ #
    #  7. Build attention mask                                            #
    # ------------------------------------------------------------------ #
    # Rule (from DINO paper):
    #  - DN slots within the same CDN group can attend to each other.
    #  - DN slots from different groups cannot attend to each other.
    #  - Matching slots can attend to each other freely.
    #  - DN slots and matching slots cannot attend to each other.
    #
    # We build a float additive mask of shape (N_total, N_total) where
    # N_total = num_dn + num_queries.
    # 0.0 = allow attention,  -inf = block attention.

    N_total = num_dn + num_queries
    attn_mask = torch.zeros(N_total, N_total, device=device)

    # Matching queries must not read GT-derived denoising queries. Denoising
    # queries are allowed to read matching queries, as in the official mask.
    attn_mask[num_dn:, :num_dn] = float("-inf")   # matching → DN: blocked

    # Block: DN slots from different groups
    group_size = 2 * max_gt  # pos + neg slots per group
    # Block all DN-DN pairs where groups differ, then restore each group.
    attn_mask[:num_dn, :num_dn] = float("-inf")
    # Restore intra-group attention to 0 (allow)
    for g in range(num_dn_groups):
        start_g = g * group_size
        end_g   = min(start_g + group_size, num_dn)
        attn_mask[start_g:end_g, start_g:end_g] = 0.0

    # ------------------------------------------------------------------ #
    #  8. Build gt_indices: for each image, map dn_slot_i → gt_box_j     #
    # ------------------------------------------------------------------ #
    # We only track positive DN slots; the loss only supervises those.
    # Shape per image: (num_dn_groups * n_i,)  pointing to GT indices.
    gt_indices: List[torch.Tensor] = []
    for i in range(batch_size):
        n_i = gt_boxes_list[i].shape[0]
        if n_i == 0:
            gt_indices.append(torch.zeros(0, dtype=torch.long, device=device))
            continue
        # Positive slots for image i are at positions:
        #   [g*2*max_gt .. g*2*max_gt + n_i - 1]  for g in 0..num_dn_groups-1
        idx = torch.cat([
            torch.arange(n_i, device=device)
            for _ in range(num_dn_groups)
        ])  # length = num_dn_groups * n_i
        gt_indices.append(idx)

    return CDNTargets(
        dn_content=dn_content,       # (B, N_dn, d_model)
        dn_ref_points=dn_ref_points, # (B, N_dn, 4)
        dn_target_boxes=dn_target_boxes,
        dn_labels=dn_labels,         # (B, N_dn)
        dn_target_labels=dn_target_labels,
        valid_slot_mask=valid_slot_mask,
        attn_mask=attn_mask,         # (N_total, N_total)
        num_dn=num_dn,
        num_groups=num_dn_groups,
        max_gt=max_gt,
        gt_indices=gt_indices,
        pos_neg_flag=pos_neg_flag,   # (B, N_dn)
    )


# ---------------------------------------------------------------------------
# CDN loss helper  (used by DINOFusionForObjectDetection.forward)
# ---------------------------------------------------------------------------

def compute_cdn_loss(
    dn_hidden_states: torch.Tensor,        # (B, num_decoder_layers, N_dn, d_model)
    dn_reference_points: torch.Tensor,     # (B, num_decoder_layers, N_dn, 4)
    dn_init_reference: torch.Tensor,       # (B, N_dn, 4)
    cdn_targets: CDNTargets,
    class_embed,                           # nn.ModuleList – one linear per decoder layer
    bbox_embed,                            # nn.ModuleList – one MLP per decoder layer
    num_classes: int,
    loss_coef_class: float = 1.0,
    loss_coef_bbox: float = 5.0,
    loss_coef_giou: float = 2.0,
) -> dict:
    """
    Compute CDN auxiliary loss on the denoising slots.

    Positive copies reconstruct their original label and box. Contrastive
    negative copies are classification background and receive no box loss.

    Returns a dict of scalar losses:
        "loss_cdn_class", "loss_cdn_bbox", "loss_cdn_giou"
    """
    _, num_layers, _, _ = dn_hidden_states.shape
    total_class_loss = dn_hidden_states.new_zeros(())
    total_bbox_loss = dn_hidden_states.new_zeros(())
    total_giou_loss = dn_hidden_states.new_zeros(())
    valid = cdn_targets.valid_slot_mask
    positive = valid & cdn_targets.pos_neg_flag.bool()
    num_positive = max(int(positive.sum().item()), 1)

    for layer_idx in range(num_layers):
        ref = (
            dn_init_reference
            if layer_idx == 0
            else dn_reference_points[:, layer_idx - 1]
        )
        hs = dn_hidden_states[:, layer_idx]
        pred_logits = class_embed[layer_idx](hs)
        delta_bbox = bbox_embed[layer_idx](hs)
        pred_boxes = (delta_bbox + _inverse_sigmoid(ref)).sigmoid()

        pred_positive = pred_boxes[positive]
        target_boxes = cdn_targets.dn_target_boxes[positive]
        logits_valid = pred_logits[valid]
        target_onehot = torch.zeros_like(logits_valid)
        # `valid` flattens slots in row-major order. Select the positive rows
        # inside that flattened subset; all other valid rows remain background.
        positive_within_valid = cdn_targets.pos_neg_flag.bool()[valid]
        positive_labels = cdn_targets.dn_target_labels[positive]
        target_onehot[
            positive_within_valid, positive_labels
        ] = 1.0

        # DINO attaches a denoising auxiliary loss to every decoder layer.
        # Classification sees positive and contrastive-negative valid slots;
        # regression sees positives only. Padding contributes to neither.
        total_class_loss += _sigmoid_focal_loss(
            logits_valid, target_onehot, num_boxes=num_positive
        )
        total_bbox_loss += F.l1_loss(
            pred_positive, target_boxes, reduction="sum"
        ) / num_positive
        total_giou_loss += (
            1.0 - _box_iou_union(pred_positive, target_boxes)
        ).sum() / num_positive

    return {
        "loss_cdn_class": loss_coef_class * total_class_loss,
        "loss_cdn_bbox": loss_coef_bbox * total_bbox_loss,
        "loss_cdn_giou": loss_coef_giou * total_giou_loss,
    }


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Sigmoid focal loss used by Deformable DETR.
    Matches HF implementation; normalised by num_boxes.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def _box_iou_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU (GIoU) between two sets of boxes in cx,cy,w,h format.
    Both tensors shape (N, 4).  Returns (N,) GIoU values.
    """
    # Convert to xyxy
    def to_xyxy(b):
        return torch.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], dim=-1)

    b1 = to_xyxy(boxes1)
    b2 = to_xyxy(boxes2)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area + 1e-6

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(b1[..., 0], b2[..., 0])
    enc_y1 = torch.min(b1[..., 1], b2[..., 1])
    enc_x2 = torch.max(b1[..., 2], b2[..., 2])
    enc_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    return giou


# ---------------------------------------------------------------------------
# Unit tests – run with:  python dino_cdn.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    print("=" * 60)
    print("dino_cdn.py – unit tests")
    print("=" * 60)

    B, D, Q, NC = 2, 256, 300, 1
    NUM_DN_GROUPS = 5

    label_emb = torch.nn.Embedding(NC + 1, D)      # +1 for "no-object" sentinel

    # ------ Test 1: normal batch with GT ------
    targets = [
        {
            "class_labels": torch.tensor([0, 0, 0]),
            "boxes": torch.tensor([[0.3, 0.4, 0.1, 0.2],
                                   [0.6, 0.7, 0.2, 0.1],
                                   [0.5, 0.5, 0.3, 0.3]]),
        },
        {
            "class_labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
        },
    ]

    cdn = build_cdn_queries(
        targets,
        label_embeddings=label_emb,
        num_dn_groups=NUM_DN_GROUPS,
        label_noise_prob=0.5,
        box_noise_scale=1.0,
        num_queries=Q,
        d_model=D,
        num_classes=NC,
    )

    max_gt = 3   # max across batch
    N_dn   = 2 * NUM_DN_GROUPS * max_gt   # 30
    N_tot  = N_dn + Q                      # 330

    assert cdn is not None,                              "CDN should not be None for non-empty batch"
    assert cdn.dn_content.shape    == (B, N_dn, D),     f"dn_content shape wrong: {cdn.dn_content.shape}"
    assert cdn.dn_ref_points.shape == (B, N_dn, 4),     f"dn_ref_points shape wrong: {cdn.dn_ref_points.shape}"
    assert cdn.dn_target_boxes.shape == (B, N_dn, 4),   f"dn_target_boxes shape wrong: {cdn.dn_target_boxes.shape}"
    assert cdn.attn_mask.shape     == (N_tot, N_tot),   f"attn_mask shape wrong: {cdn.attn_mask.shape}"
    assert cdn.pos_neg_flag.shape  == (B, N_dn),        f"pos_neg_flag shape wrong: {cdn.pos_neg_flag.shape}"
    assert cdn.num_dn              == N_dn,             "num_dn wrong"
    assert (cdn.dn_ref_points >= 0).all() and (cdn.dn_ref_points <= 1).all(), "ref_points out of [0,1]"

    # Attention mask structure checks
    # Denoising queries may read matching queries.
    assert (cdn.attn_mask[:N_dn, N_dn:] == 0.0).all(), "DN→matching must be allowed"
    # matching → DN block must be all -inf
    assert (cdn.attn_mask[N_dn:, :N_dn] == float("-inf")).all(), "matching→DN must be -inf"
    # matching → matching must be all 0
    assert (cdn.attn_mask[N_dn:, N_dn:] == 0.0).all(), "matching→matching must be 0"
    # Intra-group DN attention must be 0
    group_size = 2 * max_gt
    for g in range(NUM_DN_GROUPS):
        s = g * group_size
        e = s + group_size
        block = cdn.attn_mask[s:e, s:e]
        assert (block == 0.0).all(), f"intra-group {g} DN block must be 0"
    # Cross-group DN attention must be -inf
    s0, e0 = 0, group_size
    s1, e1 = group_size, 2 * group_size
    assert (cdn.attn_mask[s0:e0, s1:e1] == float("-inf")).all(), "cross-group must be -inf"

    print("✅ Test 1 passed – shapes and attention mask structure correct")

    # ------ Test 2: empty batch ------
    empty_targets = [
        {"class_labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
        {"class_labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
    ]
    cdn_empty = build_cdn_queries(
        empty_targets,
        label_embeddings=label_emb,
        num_dn_groups=NUM_DN_GROUPS,
        num_queries=Q,
        d_model=D,
        num_classes=NC,
    )
    assert cdn_empty is None, "CDN should be None for all-empty batch"
    print("✅ Test 2 passed – empty batch returns None")

    # ------ Test 3: pos_neg_flag layout ------
    # For NUM_DN_GROUPS=5, max_gt=3, group_size=6
    # Positive slots: [0..2], [6..8], [12..14], [18..20], [24..26]
    for g in range(NUM_DN_GROUPS):
        start = g * group_size
        assert (cdn.pos_neg_flag[0, start:start + max_gt] == 1.0).all(), \
            f"group {g} positive slots should be 1"
        assert (cdn.pos_neg_flag[0, start + max_gt:start + group_size] == 0.0).all(), \
            f"group {g} negative slots should be 0"
    print("✅ Test 3 passed – pos_neg_flag layout correct")

    # ------ Test 4: GIoU utility ------
    b1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    b2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    giou_same = _box_iou_union(b1, b2)
    assert abs(giou_same.item() - 1.0) < 1e-4, f"GIoU of identical boxes should be ~1, got {giou_same}"

    b3 = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    b4 = torch.tensor([[1.0, 1.0, 0.1, 0.1]])
    giou_far = _box_iou_union(b3, b4)
    assert giou_far.item() < 0.0, "GIoU of non-overlapping far boxes should be < 0"
    print("✅ Test 4 passed – GIoU utility correct")

    # ------ Test 5: contrastive negatives never receive box regression ------
    num_layers = 2
    class_heads = torch.nn.ModuleList(
        [torch.nn.Linear(D, NC) for _ in range(num_layers)]
    )
    bbox_heads = torch.nn.ModuleList(
        [torch.nn.Linear(D, 4) for _ in range(num_layers)]
    )
    for head in bbox_heads:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    dn_hidden = torch.randn(B, num_layers, N_dn, D)
    dn_refs = cdn.dn_ref_points[:, None].expand(-1, num_layers, -1, -1).clone()
    loss_before = compute_cdn_loss(
        dn_hidden, dn_refs, cdn.dn_ref_points, cdn,
        class_heads, bbox_heads, NC,
    )
    original_targets = cdn.dn_target_boxes.clone()
    original_target_labels = cdn.dn_target_labels.clone()
    negative_valid = cdn.valid_slot_mask & ~cdn.pos_neg_flag.bool()
    cdn.dn_target_boxes[negative_valid] = torch.rand_like(
        cdn.dn_target_boxes[negative_valid]
    )
    # A sentinel outside the class range proves that negative target labels
    # are not accidentally used as foreground classification targets.
    cdn.dn_target_labels[negative_valid] = NC + 100
    loss_after = compute_cdn_loss(
        dn_hidden, dn_refs, cdn.dn_ref_points, cdn,
        class_heads, bbox_heads, NC,
    )
    assert torch.allclose(
        loss_before["loss_cdn_class"], loss_after["loss_cdn_class"]
    ), "negative target labels must remain classification background"
    assert torch.allclose(
        loss_before["loss_cdn_bbox"], loss_after["loss_cdn_bbox"]
    ), "negative target boxes must not affect CDN L1"
    assert torch.allclose(
        loss_before["loss_cdn_giou"], loss_after["loss_cdn_giou"]
    ), "negative target boxes must not affect CDN GIoU"
    cdn.dn_target_boxes.copy_(original_targets)
    cdn.dn_target_labels.copy_(original_target_labels)
    print("✅ Test 5 passed – negative CDN slots are background without box loss")

    # ------ Test 6: CDN classification is not multiplied by query count ------
    valid = cdn.valid_slot_mask
    positive = valid & cdn.pos_neg_flag.bool()
    num_positive = max(int(positive.sum().item()), 1)
    expected_class = dn_hidden.new_zeros(())
    for layer_idx in range(num_layers):
        logits_valid = class_heads[layer_idx](dn_hidden[:, layer_idx])[valid]
        target_onehot = torch.zeros_like(logits_valid)
        positive_within_valid = cdn.pos_neg_flag.bool()[valid]
        target_onehot[
            positive_within_valid, cdn.dn_target_labels[positive]
        ] = 1.0
        expected_class += _sigmoid_focal_loss(
            logits_valid, target_onehot, num_boxes=num_positive
        )
    assert torch.allclose(
        loss_before["loss_cdn_class"], expected_class
    ), "CDN focal loss must not be multiplied by the number of DN queries"
    print("✅ Test 6 passed – CDN classification has the correct normalization")

    print()
    print("All tests passed ✅")
    sys.exit(0)
