"""Isolated Ultralytics YOLO26 RGB+IR integration.

This package intentionally does not import :mod:`sarfusion.models` or
:mod:`sarfusion.data`: those packages depend on the historical THU-MIG
YOLOv10 fork, whereas this integration targets upstream Ultralytics 8.4.138.
"""

from .model import YOLO26FusionDetectionModel

__all__ = ["YOLO26FusionDetectionModel"]
