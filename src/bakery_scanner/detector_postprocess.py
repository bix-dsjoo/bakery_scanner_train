from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .detector_evaluation import Detection
from .errors import DataValidationError


@dataclass(frozen=True, slots=True)
class DetectorPostprocessConfig:
    confidence_threshold: float
    nms_iou: float
    max_symmetric_aspect_ratio: float

    def validate(self) -> None:
        values = (
            self.confidence_threshold,
            self.nms_iou,
            self.max_symmetric_aspect_ratio,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise DataValidationError("detector postprocess values must be finite numbers")
        if not 0 < self.confidence_threshold < 1:
            raise DataValidationError("confidence_threshold must be in (0, 1)")
        if not 0 < self.nms_iou <= 1:
            raise DataValidationError("nms_iou must be in (0, 1]")
        if self.max_symmetric_aspect_ratio <= 1:
            raise DataValidationError("max_symmetric_aspect_ratio must be greater than 1")


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _normalized_detection(value: Detection, index: int) -> Detection:
    if not isinstance(value, Detection):
        raise DataValidationError(f"detection {index} has invalid type")
    if value.class_index != 0:
        raise DataValidationError(f"detection {index} must use class 0")
    if (
        isinstance(value.confidence, bool)
        or not isinstance(value.confidence, (int, float))
        or not math.isfinite(value.confidence)
        or not 0 <= value.confidence <= 1
    ):
        raise DataValidationError(f"detection {index} confidence is invalid")
    if len(value.bbox_xyxy) != 4 or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value.bbox_xyxy
    ):
        raise DataValidationError(f"detection {index} bbox is invalid")
    x1, y1, x2, y2 = (float(item) for item in value.bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise DataValidationError(f"detection {index} bbox must have positive area")
    return Detection((x1, y1, x2, y2), float(value.confidence), 0)


def filter_detections(
    detections: Iterable[Detection], config: DetectorPostprocessConfig
) -> tuple[Detection, ...]:
    if not isinstance(config, DetectorPostprocessConfig):
        raise DataValidationError("config must be DetectorPostprocessConfig")
    config.validate()
    eligible: list[Detection] = []
    for index, raw_detection in enumerate(detections):
        detection = _normalized_detection(raw_detection, index)
        if detection.confidence < config.confidence_threshold:
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        if max(width / height, height / width) > config.max_symmetric_aspect_ratio:
            continue
        eligible.append(detection)

    kept: list[Detection] = []
    for detection in sorted(
        eligible, key=lambda item: (-item.confidence, item.bbox_xyxy)
    ):
        if all(
            _iou(detection.bbox_xyxy, previous.bbox_xyxy) < config.nms_iou
            for previous in kept
        ):
            kept.append(detection)
    return tuple(kept)
