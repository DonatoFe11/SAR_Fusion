import torch
from sarfusion.utils.utils import EasyDict


class ModelOutput(EasyDict):
    features: list[torch.Tensor]
    features_aux: list[torch.Tensor]
    logits: list[torch.Tensor]
    logits_aux: list[torch.Tensor]
    predictions: list[torch.Tensor]


class LossOutput(EasyDict):
    value: torch.Tensor
    components: dict


class WrapperModelOutput(ModelOutput):
    loss: LossOutput


class DataDict(EasyDict):
    pixel_values: torch.Tensor
    labels: torch.Tensor
    pixel_mask: torch.Tensor
    dims: tuple
    path: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "pixel_values" not in self:
            self.pixel_values = None
        if "labels" not in self:
            self.labels = None