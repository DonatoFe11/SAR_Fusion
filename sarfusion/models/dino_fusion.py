from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrMultiheadAttention,
    DeformableDetrDecoderOutput,
    DeformableDetrModelOutput,
    DeformableDetrHungarianMatcher,
    DeformableDetrLoss,
    DeformableDetrObjectDetectionOutput,
    inverse_sigmoid,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deformable_detr.configuration_deformable_detr import (
    DeformableDetrConfig,
)

from sarfusion.models.deformable_detr_fusion import (
    DeformableDetrFusionForObjectDetection,
    DeformableDetrFusionModel,
)
from sarfusion.models.dino_cdn import (
    CDNTargets,
    build_cdn_queries,
    compute_cdn_loss,
)


# ---------------------------------------------------------------------------
# 1. Decoder: square CDN mask + Look-Forward-Twice
# ---------------------------------------------------------------------------

class DINODeformableDetrMultiheadAttention(DeformableDetrMultiheadAttention):
    """HF self-attention accepting DINO's additive square query mask."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=False,
    ):
        batch_size, target_len, embed_dim = hidden_states.size()
        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    "DINO decoder attention mask must be (batch, 1, query_length, query_length)"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights_reshaped = (
            attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            if output_attentions
            else None
        )
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)
        return self.out_proj(attn_output), attn_weights_reshaped


class DINODeformableDetrDecoderLayer(DeformableDetrDecoderLayer):
    """HF decoder layer extended with DINO's square self-attention mask."""

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.self_attn = DINODeformableDetrMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions=False,
    ):
        residual = hidden_states
        attention_mask = None
        if decoder_attention_mask is not None:
            # DeformableDetrMultiheadAttention consumes an additive 4-D mask:
            # (batch, 1, target_length, source_length).
            if decoder_attention_mask.dim() == 2:
                attention_mask = decoder_attention_mask[None, None].expand(
                    hidden_states.shape[0], -1, -1, -1
                )
            elif decoder_attention_mask.dim() == 3:
                attention_mask = decoder_attention_mask[:, None]
            else:
                raise ValueError("decoder_attention_mask must have 2 or 3 dimensions")
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.training and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs


class LFTDecoder(DeformableDetrDecoder):
    """DINO decoder with active LFT and a functional CDN attention mask."""

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DINODeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)]
        )

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = inputs_embeds
        intermediate = ()
        intermediate_reference_points = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1
                )[:, None]
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError(f"Last dim of reference_points must be 2 or 4, got {num_coordinates}")

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if self.config.with_box_refine:
                tmp = self.bbox_embed[layer_idx](hidden_states)
                if num_coordinates == 4:
                    new_reference_points = (tmp + inverse_sigmoid(reference_points)).sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # DINO Look-Forward-Twice keeps the refined box in the graph for
                # the prediction made from the next decoder output, but detaches
                # the reference fed to the next attention layer.  Consequently
                # layer i receives gradients from predictions i and i+1 without
                # backpropagating through every later deformable-attention block.
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (new_reference_points,)
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                value
                for value in [
                    hidden_states,
                    torch.stack(intermediate, dim=1),
                    torch.stack(intermediate_reference_points, dim=1),
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if value is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=torch.stack(intermediate, dim=1),
            intermediate_reference_points=torch.stack(intermediate_reference_points, dim=1),
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# ---------------------------------------------------------------------------
# 2. Inner model — two-stage anchors + mixed query selection + LFT
# ---------------------------------------------------------------------------

class DINOFusionModel(DeformableDetrFusionModel):
    """Fusion model implementing the DINO query initialisation path."""

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
        if not config.two_stage or not config.with_box_refine:
            raise ValueError("DINOFusionModel requires two_stage=True and with_box_refine=True")
        self.decoder = LFTDecoder(config)
        # DINO mixed query selection: encoder proposals provide the anchors,
        # while the decoder content comes from independent learned queries.
        self.mixed_query_content = nn.Embedding(config.num_queries, config.d_model)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        cdn_targets: Optional[CDNTargets] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is not None or decoder_inputs_embeds is not None:
            raise ValueError("DINOFusionModel builds its decoder queries internally")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = pixel_values.shape[0]
        if pixel_mask is None:
            pixel_mask = torch.ones(
                pixel_values.shape[0], pixel_values.shape[-2], pixel_values.shape[-1],
                dtype=torch.bool, device=pixel_values.device,
            )

        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)
        sources, masks = [], []
        for level, feature in enumerate(features):
            source, mask = feature
            if mask is None:
                raise ValueError("The fusion backbone must return a pixel mask for every feature level")
            sources.append(self.input_proj[level](source))
            masks.append(mask)
        for level in range(len(sources), self.config.num_feature_levels):
            source = self.input_proj[level](features[-1][0] if level == len(features) else sources[-1])
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
            pos_embed = self.backbone.position_embedding(source, mask).to(source.dtype)
            sources.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_embed)

        source_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            _, _, height, width = source.shape
            spatial_shapes.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed + self.level_embed[level].view(1, 1, -1))
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(mask, dtype=source_flatten.dtype) for mask in masks], 1
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
            encoder_outputs[0], ~mask_flatten, spatial_shapes
        )
        enc_outputs_class = self.decoder.class_embed[-1](object_query_embedding)
        enc_outputs_coord_logits = self.decoder.bbox_embed[-1](object_query_embedding) + output_proposals
        # DINO ranks encoder proposals by their best foreground class.  Using
        # channel zero happens to work for WiSARD's single class, but silently
        # breaks mixed query selection as soon as the model is multi-class.
        topk_scores = enc_outputs_class.max(dim=-1).values
        topk_indices = torch.topk(topk_scores, self.config.num_queries, dim=1).indices
        topk_coords_logits = torch.gather(
            enc_outputs_coord_logits, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        ).detach()
        matching_reference_points = topk_coords_logits.sigmoid()
        matching_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_logits)))
        matching_pos, _ = torch.split(matching_pos, self.config.d_model, dim=2)
        matching_target = self.mixed_query_content.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if cdn_targets is not None:
            dn_logits = inverse_sigmoid(cdn_targets.dn_ref_points)
            dn_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(dn_logits)))
            dn_pos, _ = torch.split(dn_pos, self.config.d_model, dim=2)
            reference_points = torch.cat([cdn_targets.dn_ref_points, matching_reference_points], dim=1)
            position_embeddings = torch.cat([dn_pos, matching_pos], dim=1)
            target = torch.cat([cdn_targets.dn_content, matching_target], dim=1)
        else:
            reference_points = matching_reference_points
            position_embeddings = matching_pos
            target = matching_target

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            enc_outputs = tuple(
                value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None
            )
            return (reference_points,) + decoder_outputs + encoder_outputs + enc_outputs
        return DeformableDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            init_reference_points=reference_points,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


# ---------------------------------------------------------------------------
# 3.  Detection head — CDN injection + CDN loss
# ---------------------------------------------------------------------------

class DINOFusionForObjectDetection(DeformableDetrFusionForObjectDetection):
    """
    Full DINO-style RGB-IR fusion model.

    Adds to DeformableDetrFusionForObjectDetection all three DINO mechanisms:
      • contrastive denoising (CDN), including its decoder attention mask;
      • mixed query selection (encoder anchors + learned query content);
      • look-forward-twice (LFT) with iterative box refinement.

    Extra constructor parameters (all have sensible defaults):
      num_dn_groups    : int   = 5     number of CDN groups
      label_noise_prob : float = 0.5   DINO label-noise ratio
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
        # The base constructor attached these heads to the model it created.
        # Re-attach them after replacing it, otherwise LFT and two-stage
        # proposal selection would silently have no prediction heads.
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.num_dn_groups    = num_dn_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale  = box_noise_scale
        self.cdn_loss_coef    = cdn_loss_coef

        # Dedicated nn.Embedding for CDN content vectors.
        # Maps class index (0..num_classes) → d_model vector.
        # Size is num_classes+1 to include the "no-object" sentinel for negative DN slots.
        # Must NOT be class_embed, which is a Linear(d_model → num_classes+1).
        self.dn_label_embeddings = nn.Embedding(
            len(config.id2label) + 1, config.d_model
        )

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
                label_embeddings=self.dn_label_embeddings,
                num_dn_groups=self.num_dn_groups,
                label_noise_prob=self.label_noise_prob,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.config.num_queries,
                d_model=self.config.d_model,
                num_classes=len(self.config.id2label),
                device=pixel_values.device,
            )

        # The inner model derives positional embeddings from the selected
        # two-stage anchors; CDN reference boxes use that same DINO path.
        attn_mask_for_decoder = cdn_targets.attn_mask if cdn_targets is not None else decoder_attention_mask
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=attn_mask_for_decoder,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            cdn_targets=cdn_targets,
            **kwargs,
        )

        # Split denoising slots before applying the regular detection heads.
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
            matcher = DeformableDetrHungarianMatcher(
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
            )
            losses = ["labels", "boxes", "cardinality"]
            criterion = DeformableDetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            )
            criterion.to(self.device)

            outputs_loss = {"logits": logits, "pred_boxes": pred_boxes}
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            if self.config.two_stage:
                enc_outputs_coord = outputs.enc_outputs_coord_logits.sigmoid()
                outputs_loss["enc_outputs"] = {
                    "logits": outputs.enc_outputs_class,
                    "pred_boxes": enc_outputs_coord,
                }

            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {
                "loss_ce": 1,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient,
            }
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            if self.config.two_stage:
                # The selected anchors are trained by the encoder proposal loss.
                enc_weight_dict = {
                    "loss_ce_enc": 1,
                    "loss_bbox_enc": self.config.bbox_loss_coefficient,
                    "loss_giou_enc": self.config.giou_loss_coefficient,
                }
                weight_dict.update(enc_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
                if loss_dict is None:
                    loss_dict = {}
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
        # The public checkpoint is one-stage Deformable DETR.  A real DINO
        # decoder instead requires encoder proposals and iterative refinement.
        config.two_stage = True
        config.two_stage_num_proposals = config.num_queries
        config.with_box_refine = True

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
