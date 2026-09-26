"""RGB+IR integration for Ultralytics YOLO26 8.4.138.

Keep imports independent of sarfusion.models and sarfusion.data, which
use the THU-MIG YOLOv10 fork."""

from .model import YOLO26FusionDetectionModel

__all__ = ["YOLO26FusionDetectionModel"]
