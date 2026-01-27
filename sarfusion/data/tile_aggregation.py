"""
Utility per aggregare predizioni da sliding window inference (tiling).

Questo modulo fornisce funzioni per combinare le predizioni di più tile
di un'immagine in un'unica predizione aggregata.
"""

import torch
from collections import defaultdict
from torchvision.ops import nms


def xywh_to_xyxy(boxes):
    """
    Converte bounding boxes da formato xywh (centro) a xyxy (corners).
    
    Args:
        boxes: Tensor [N, 4] con box in formato [cx, cy, w, h]
    
    Returns:
        boxes_xyxy: Tensor [N, 4] con box in formato [x1, y1, x2, y2]
    """
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return boxes_xyxy


def remap_tile_boxes_to_original(boxes, quadrant, tile_size=320, orig_size=640):
    """
    Rimappa le coordinate delle bounding box dal tile all'immagine originale.
    
    Args:
        boxes: Tensor [N, 4] con box in formato [cx, cy, w, h] (xywh - centro)
               **NORMALIZZATE** relative al tile (coordinate 0-1)
        quadrant: int 0-3 indicante il quadrante
                  0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        tile_size: dimensione del tile (default 320)
        orig_size: dimensione immagine originale (default 640)
    
    Returns:
        boxes_remapped: Tensor [N, 4] con coordinate normalizzate nell'immagine originale
    """
    scale_factor = tile_size / orig_size  # 320/640 = 0.5
    
    # Offset normalizzato (0 o 0.5)
    x_offset_norm = (quadrant % 2) * scale_factor   # 0 o 0.5
    y_offset_norm = (quadrant // 2) * scale_factor  # 0 o 0.5
    
    boxes_remapped = boxes.clone()
    # Scala e trasla cx, cy
    boxes_remapped[:, 0] = boxes[:, 0] * scale_factor + x_offset_norm  # cx
    boxes_remapped[:, 1] = boxes[:, 1] * scale_factor + y_offset_norm  # cy
    # Scala w, h
    boxes_remapped[:, 2] = boxes[:, 2] * scale_factor  # w
    boxes_remapped[:, 3] = boxes[:, 3] * scale_factor  # h
    
    return boxes_remapped


def aggregate_tile_predictions(tile_predictions, iou_threshold=0.5):
    """
    Aggrega le predizioni di tutti i tile di un'immagine con NMS globale.
    
    Args:
        tile_predictions: Lista di dict, uno per tile, con chiavi:
                          - 'boxes': Tensor [N, 4] in coordinate del tile
                          - 'scores': Tensor [N]
                          - 'labels': Tensor [N]
                          - 'quadrant': int
        iou_threshold: soglia IoU per NMS
    
    Returns:
        dict con chiavi 'boxes', 'scores', 'labels' aggregate
    """
    if not tile_predictions:
        return {
            'boxes': torch.zeros((0, 4)),
            'scores': torch.zeros(0),
            'labels': torch.zeros(0, dtype=torch.long)
        }
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for tile_pred in tile_predictions:
        quadrant = tile_pred['quadrant']
        boxes = tile_pred['boxes']
        scores = tile_pred['scores']
        labels = tile_pred['labels']
        
        # Rimappa le coordinate
        boxes_remapped = remap_tile_boxes_to_original(boxes, quadrant)
        
        all_boxes.append(boxes_remapped)
        all_scores.append(scores)
        all_labels.append(labels)
    
    # Concatena tutte le predizioni
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if len(all_boxes) == 0:
        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'labels': all_labels
        }
    
    # Converti xywh -> xyxy per NMS
    all_boxes_xyxy = xywh_to_xyxy(all_boxes)
    
    # NMS globale per rimuovere duplicati (richiede formato xyxy)
    keep_indices = nms(all_boxes_xyxy, all_scores, iou_threshold)
    
    # Ritorna in formato xywh (come le predizioni originali)
    return {
        'boxes': all_boxes[keep_indices],
        'scores': all_scores[keep_indices],
        'labels': all_labels[keep_indices]
    }


def group_batch_by_original_image(batch_dict, result_dict):
    """
    Raggruppa i risultati di un batch per immagine originale.
    
    Utile quando il batch contiene più tile della stessa immagine.
    
    Args:
        batch_dict: dizionario con metadati, deve contenere 'original_idx'
        result_dict: risultati del modello con predizioni per ogni tile
    
    Returns:
        dict: {original_idx: [list of predictions for that image]}
    """
    grouped = defaultdict(list)
    
    batch_size = len(batch_dict.get('original_idx', []))
    
    for i in range(batch_size):
        original_idx = batch_dict['original_idx'][i].item()
        quadrant = batch_dict['quadrant'][i].item()
        
        tile_pred = {
            'boxes': result_dict.boxes[i],
            'scores': result_dict.scores[i],
            'labels': result_dict.labels[i],
            'quadrant': quadrant,
        }
        
        grouped[original_idx].append(tile_pred)
    
    return grouped


# Esempio di utilizzo nell'evaluation loop:
"""
def evaluate_with_tiling(self, dataloader, epoch=None, phase="val"):
    '''
    Valutazione con aggregazione delle predizioni dei tile.
    '''
    from sarfusion.data.tile_aggregation import (
        group_batch_by_original_image,
        aggregate_tile_predictions
    )
    
    self.model.eval()
    self.val_evaluator.reset()
    
    # Buffer per accumulare predizioni di tutti i tile di ogni immagine
    image_predictions_buffer = {}
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            batch_dict = DataDict(**batch_dict)
            result_dict = self.model(batch_dict)
            
            # Raggruppa per immagine originale
            grouped_preds = group_batch_by_original_image(batch_dict, result_dict)
            
            # Accumula nel buffer
            for orig_idx, tile_preds in grouped_preds.items():
                if orig_idx not in image_predictions_buffer:
                    image_predictions_buffer[orig_idx] = []
                image_predictions_buffer[orig_idx].extend(tile_preds)
                
                # Se abbiamo tutti i 4 tile, aggrega e valuta
                if len(image_predictions_buffer[orig_idx]) == 4:
                    aggregated = aggregate_tile_predictions(
                        image_predictions_buffer[orig_idx],
                        iou_threshold=0.5
                    )
                    
                    # Aggiorna metriche con predizioni aggregate
                    self._update_val_metrics_aggregated(
                        batch_dict, 
                        aggregated, 
                        orig_idx
                    )
                    
                    # Rimuovi dal buffer
                    del image_predictions_buffer[orig_idx]
    
    metrics_dict = self.val_evaluator.compute()
    return metrics_dict
"""
