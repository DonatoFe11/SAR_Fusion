"""
dino_fusion.py
--------------
DINO-style training on top of the existing RGB-IR Fusion backbone.

Class hierarchy:
    DeformableDetrForObjectDetection               (HuggingFace, untouched)
      └─ DeformableDetrFusionFAMForObjectDetection  (deformable_detr_fusion_fam.py, untouched)
           └─ DINOFusionForObjectDetection           ← this file

What this file adds vs. the parent:
  1. CDN query injection in forward()   – prepend DN slots to decoder input
  2. CDN loss in forward()              – supervised on DN slots only
  3. Look Forward Twice (LFT)           – gradient flows through reference points
     between decoder layers via LFTDecoder

Place this file at:
    sarfusion/models/dino_fusion.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderOutput,
    DeformableDetrObjectDetectionOutput,
    inverse_sigmoid,
)
from transformers.models.deformable_detr.configuration_deformable_detr import (
    DeformableDetrConfig,
)

from sarfusion.models.deformable_detr_fusion_fam import (
    DeformableDetrFusionFAMForObjectDetection,
    DeformableDetrFusionFAMModel,
)
from sarfusion.models.dino_cdn import (
    CDNTargets,
    build_cdn_queries,
    compute_cdn_loss,
)


# ---------------------------------------------------------------------------
# 1.  Look-Forward-Twice decoder
#     The only change vs. the HF decoder: reference_points are NOT detached
#     between layers, so gradients flow backwards through box predictions.
# ---------------------------------------------------------------------------

class LFTDecoder(DeformableDetrDecoder):
    """
    Deformable DETR decoder with Look-Forward-Twice (LFT).

    The original HF decoder does:
        reference_points = new_reference_points.detach()
    which cuts gradients between layers.  Removing .detach() is the entirety
    of LFT; everything else is identical to the parent.
    """

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries_position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions or self.config.output_attentions

        hidden_states = inputs_embeds
        intermediate = ()
        intermediate_reference_points = ()

        for decoder_layer in self.layers:
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif num_coordinates == 2:
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )
            else:
                raise ValueError(
                    f"Last dim of reference_points must be 2 or 4, got {num_coordinates}"
                )

            layer_outputs = decoder_layer(
                hidden_states,
                object_queries_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            # Box refinement (same as HF original)
            if self.config.with_box_refine:
                tmp = self.bbox_embed[len(intermediate)](hidden_states)
                if num_coordinates == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = (
                        tmp[..., :2] + inverse_sigmoid(reference_points)
                    )
                    new_reference_points = new_reference_points.sigmoid()
                # LFT: NO .detach() — gradient flows back through reference points
                reference_points = new_reference_points

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )


# ---------------------------------------------------------------------------
# 2.  Inner model — swaps in LFTDecoder, keeps fusion backbone
# ---------------------------------------------------------------------------

class DINOFusionModel(DeformableDetrFusionFAMModel):
    """
    Replaces the standard decoder with LFTDecoder.
    Backbone and encoder are fully inherited.
    """

    def __init__(
        self,
        config: DeformableDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
    ):
        super().__init__(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )
        self.decoder = LFTDecoder(config)


# ---------------------------------------------------------------------------
# 3.  Detection head — CDN injection + CDN loss
# ---------------------------------------------------------------------------

class DINOFusionForObjectDetection(DeformableDetrFusionFAMForObjectDetection):
    """
    Full DINO-style RGB-IR fusion model.

    Adds to DeformableDetrFusionFAMForObjectDetection:
      • CDN query injection (training only)
      • CDN loss on denoising slots
      • Look-Forward-Twice via LFTDecoder

    Extra constructor parameters (all have sensible defaults):
      num_dn_groups    : int   = 5     number of CDN groups
      label_noise_prob : float = 0.5   probability of label flip for positive DN slots
      box_noise_scale  : float = 1.0   noise scale (positive slots get /2, negative full)
      cdn_loss_coef    : float = 1.0   multiplier on the total CDN loss term
    """

    def __init__(
        self,
        config: DeformableDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        num_dn_groups: int = 5,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
        cdn_loss_coef: float = 1.0,
    ):
        super().__init__(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )
        # Replace inner model with DINO variant (LFT decoder)
        self.model = DINOFusionModel(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )
        self.num_dn_groups    = num_dn_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale  = box_noise_scale
        self.cdn_loss_coef    = cdn_loss_coef
        self.post_init()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[Dict]] = None,
        **kwargs,
    ):
        # ----------------------------------------------------------------
        # Step A: build CDN queries (training only, only when labels given)
        # ----------------------------------------------------------------
        cdn_targets: Optional[CDNTargets] = None
        if self.training and labels is not None:
            cdn_targets = build_cdn_queries(
                targets=labels,
                label_embeddings=self.class_embed[-1],
                pos_embeddings=self.model.query_position_embeddings,
                num_dn_groups=self.num_dn_groups,
                label_noise_prob=self.label_noise_prob,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.config.num_queries,
                d_model=self.config.d_model,
                num_classes=len(self.config.id2label),
                device=pixel_values.device,
            )

        # ----------------------------------------------------------------
        # Step B: temporarily augment query_position_embeddings with DN rows
        #
        # DeformableDetrModel.forward() does internally:
        #   query_embed, target = split(self.query_position_embeddings.weight, d_model, dim=1)
        #   reference_points = self.reference_points(query_embed).sigmoid()
        #   decoder(inputs_embeds=target, object_queries_position_embeddings=query_embed, ...)
        #
        # By prepending DN rows to the weight we make the inner model treat DN
        # slots like extra learned queries — no duplication of the model forward.
        # ----------------------------------------------------------------
        original_qpe = None
        attn_mask_for_decoder = decoder_attention_mask

        if cdn_targets is not None:
            d = self.config.d_model
            N_dn = cdn_targets.num_dn

            # DN rows in (2*d_model) format: [pos_half | content_half]
            # Use image-0's vectors; they are identical across the batch for the
            # positional half; content varies but we use image-0 as the embedding
            # seed (the loss handles per-image correctness via gt_indices).
            dn_pos_half     = cdn_targets.dn_pos[0].detach()      # (N_dn, d)
            dn_content_half = cdn_targets.dn_content[0].detach()  # (N_dn, d)
            dn_rows = torch.cat([dn_pos_half, dn_content_half], dim=-1)  # (N_dn, 2d)

            original_weight = self.model.query_position_embeddings.weight.data  # (Q, 2d)
            augmented_weight = torch.cat([dn_rows, original_weight], dim=0)     # (N_dn+Q, 2d)

            original_qpe = self.model.query_position_embeddings
            aug_emb = nn.Embedding(N_dn + self.config.num_queries, 2 * d,
                                   device=pixel_values.device)
            aug_emb.weight = nn.Parameter(augmented_weight)
            self.model.query_position_embeddings = aug_emb

            # Attention mask: (N_dn+Q, N_dn+Q) float additive mask
            attn_mask_for_decoder = cdn_targets.attn_mask  # (N_total, N_total)

        # ----------------------------------------------------------------
        # Step C: run the inner model forward
        # ----------------------------------------------------------------
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=attn_mask_for_decoder,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )

        # ----------------------------------------------------------------
        # Step D: restore original query_position_embeddings
        # ----------------------------------------------------------------
        if original_qpe is not None:
            self.model.query_position_embeddings = original_qpe

        # ----------------------------------------------------------------
        # Step E: split DN slots from matching slots
        # ----------------------------------------------------------------
        N_dn = cdn_targets.num_dn if cdn_targets is not None else 0

        # shapes after inner model: (B, num_layers, N_total, d)
        hidden_states = outputs.intermediate_hidden_states        # (B, L, N_total, d)
        inter_refs    = outputs.intermediate_reference_points     # (B, L, N_total, 4)
        init_ref      = outputs.init_reference_points             # (B, N_total, 4)

        dn_hidden = dn_refs = dn_init = None
        if N_dn > 0:
            dn_hidden = hidden_states[:, :, :N_dn, :]   # (B, L, N_dn, d)
            dn_refs   = inter_refs[:, :, :N_dn, :]      # (B, L, N_dn, 4)
            dn_init   = init_ref[:, :N_dn, :]           # (B, N_dn, 4)

            hidden_states = hidden_states[:, :, N_dn:, :]  # (B, L, Q, d)
            inter_refs    = inter_refs[:, :, N_dn:, :]     # (B, L, Q, 4)
            init_ref      = init_ref[:, N_dn:, :]          # (B, Q, 4)

        # ----------------------------------------------------------------
        # Step F: class + box predictions on matching slots
        # ----------------------------------------------------------------
        outputs_classes = []
        outputs_coords  = []

        for level in range(hidden_states.shape[1]):
            reference = init_ref if level == 0 else inter_refs[:, level - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[level](hidden_states[:, level])
            delta_bbox    = self.bbox_embed[level](hidden_states[:, level])

            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord_logits.sigmoid())

        outputs_class = torch.stack(outputs_classes)  # (L, B, Q, C+1)
        outputs_coord = torch.stack(outputs_coords)   # (L, B, Q, 4)

        logits     = outputs_class[-1]   # (B, Q, C+1)
        pred_boxes = outputs_coord[-1]   # (B, Q, 4)

        # ----------------------------------------------------------------
        # Step G: loss
        # ----------------------------------------------------------------
        loss, loss_dict, auxiliary_outputs = None, None, None

        if labels is not None:
            # Standard Hungarian matching loss (inherited from HF)
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
            )

            # CDN loss (training only, only when CDN was active)
            if cdn_targets is not None and dn_hidden is not None:
                cdn_loss_dict = compute_cdn_loss(
                    dn_hidden_states=dn_hidden,
                    dn_reference_points=dn_refs,
                    dn_init_reference=dn_init,
                    cdn_targets=cdn_targets,
                    class_embed=self.class_embed,
                    bbox_embed=self.bbox_embed,
                    num_classes=len(self.config.id2label),
                )
                cdn_total = sum(cdn_loss_dict.values()) * self.cdn_loss_coef
                loss = loss + cdn_total
                loss_dict.update(cdn_loss_dict)

        return DeformableDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=hidden_states,
            intermediate_reference_points=inter_refs,
            init_reference_points=init_ref,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

    # ------------------------------------------------------------------
    # from_pretrained
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        id2label: dict,
        label2id: dict,
        ignore_mismatched_sizes: bool = True,
        num_feature_levels: Optional[int] = None,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        num_dn_groups: int = 5,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
        cdn_loss_coef: float = 1.0,
        **kwargs,
    ) -> "DINOFusionForObjectDetection":
        from transformers import DeformableDetrForObjectDetection

        base_model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs,
        )

        config = base_model.config
        if num_feature_levels is not None:
            config.num_feature_levels = num_feature_levels

        model = cls(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            num_dn_groups=num_dn_groups,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            cdn_loss_coef=cdn_loss_coef,
        )

        # Permissive weight load (backbone shapes will mismatch — that's expected)
        model.load_state_dict(base_model.state_dict(), strict=False)

        # Copy position embeddings to both RGB and IR backbones
        position_emb_state = {
            k.replace("model.backbone.", ""): v
            for k, v in base_model.state_dict().items()
            if "backbone.position_embedding" in k
        }
        model.model.backbone.rgb_backbone.position_embedding.load_state_dict(
            position_emb_state, strict=False
        )
        model.model.backbone.ir_backbone.position_embedding.load_state_dict(
            position_emb_state, strict=False
        )

        # Adapt first conv of IR backbone: average RGB channels → 1 channel
        rgb_backbone_state = base_model.model.backbone.conv_encoder.state_dict()
        ir_backbone_state = {
            k: (v.mean(dim=1, keepdim=True) if v.dim() == 4 and v.shape[1] == 3 else v)
            for k, v in rgb_backbone_state.items()
        }
        model.model.backbone.rgb_backbone.conv_encoder.load_state_dict(
            rgb_backbone_state, strict=False
        )
        model.model.backbone.ir_backbone.conv_encoder.load_state_dict(
            ir_backbone_state, strict=False
        )

        # Stable init for channel_fusion blocks
        for fusion_block in model.model.backbone.channel_fusion:
            nn.init.xavier_uniform_(fusion_block[0].weight)
            nn.init.zeros_(fusion_block[0].bias)
            nn.init.ones_(fusion_block[1].weight)
            nn.init.zeros_(fusion_block[1].bias)

        return model