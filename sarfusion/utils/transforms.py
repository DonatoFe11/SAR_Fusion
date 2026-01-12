import numpy as np

import torch
import torchvision.transforms.functional as F

from torchvision.transforms import Resize


class ResizePadKeepRatio(Resize):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, put_padding_on_right_bottom=True):
        super(ResizePadKeepRatio, self).__init__(size, interpolation)
        self.size = size
        self.put_padding_on_right_bottom = put_padding_on_right_bottom

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            _, w, h = img.size()
        else:
            w, h = img.size
        if w > h:
            new_w = self.size
            new_h = int(h * new_w / w)
        else:
            new_h = self.size
            new_w = int(w * new_h / h)
        resized = F.resize(img, (new_w, new_h), self.interpolation)
        r = self.size / max(w, h)
        dw, dh = self.size - new_w, self.size - new_h
        if self.put_padding_on_right_bottom:
            pad = (0, 0, dh, dw)
        else:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            pad = (left, top, right, bottom)
        img = F.pad(resized, pad)
        return img, r, pad
    
    
def letterbox(im, new_shape=(640, 640), scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[1:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR) 
        im = F.resize(im, new_unpad, interpolation=F.InterpolationMode.BILINEAR)
    top, bottom = int(round(dw - 0.1)), int(round(dw + 0.1))
    left, right = int(round(dh - 0.1)), int(round(dh + 0.1))
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border 
    im = F.pad(im, (left, top, right, bottom), fill=0)
    return im, ratio, (dw, dh)