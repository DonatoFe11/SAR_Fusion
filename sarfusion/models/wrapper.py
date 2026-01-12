from dataclasses import dataclass
import torch
import torch.nn as nn

from transformers.utils import ModelOutput

from sarfusion.utils.structures import DataDict


@dataclass
class WrapperModelOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None


# class ModelWrapper(nn.Module):
#     def __init__(self, model, loss):
#         super(ModelWrapper, self).__init__()
#         self.model = model
#         self.loss = loss
        
#     def forward(self, pixel_values, labels):
#         outputs = self.model(pixel_values)
#         if self.loss is None:
#             return WrapperModelOutput(logits=outputs.logits)
#         loss = self.loss(outputs.logits, labels)
#         return WrapperModelOutput(loss=loss, logits=outputs.logits)
        