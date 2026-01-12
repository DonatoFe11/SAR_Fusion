from copy import deepcopy
import torch.nn as nn
from sarfusion.experiment.utils import unwrap_model
from sarfusion.utils.loss_tal_dual import ComputeLoss as YOLOLoss
from sarfusion.utils.lossv10 import v10DetectLoss

LOSS_REGISTRY = {
    "yolo_loss": YOLOLoss,
    "v10_loss": v10DetectLoss,
}


class TorchLossWrapper(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, result_dict, target):
        logits = result_dict.logits
        return self.loss(logits, target)


def build_loss(params, model=None):
    if not params:
        return None

    name = params["name"]
    params = deepcopy(params["params"])
    if params.get("requires_model", False):
        params.pop("requires_model")
        params["model"] = unwrap_model(model)
    if hasattr(model, "loss_fn"):
        return model.loss_fn
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name](**params)
    torch_losses = nn.__dict__
    if name in torch_losses:
        return TorchLossWrapper(torch_losses[name](**params))
    raise ValueError(f"Loss {name} not found in torch.nn or LOSS_REGISTRY")
