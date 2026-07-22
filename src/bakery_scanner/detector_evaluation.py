from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import DataValidationError


@dataclass(frozen=True, slots=True)
class EvaluationObject:
    bbox_xyxy: tuple[float, float, float, float]
    category_id: int
    phase: str


@dataclass(frozen=True, slots=True)
class EvaluationImage:
    image_id: str
    difficulty: str | None
    objects: tuple[EvaluationObject, ...]


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_index: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    confidence_floor: float = 0.001
    operating_confidence: float = 0.25
    nms_iou: float = 0.7
    matching_iou: float = 0.5
    max_symmetric_aspect_ratio: float | None = None


def _validate_box(
    value: tuple[float, float, float, float], label: str
) -> tuple[float, float, float, float]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise DataValidationError(f"{label} must contain four xyxy values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise DataValidationError(f"{label} must contain finite numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise DataValidationError(f"{label} must have positive area")
    return x1, y1, x2, y2


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


def _match(
    images: Sequence[EvaluationImage],
    predictions: Mapping[str, Sequence[Detection]],
    minimum_confidence: float,
    matching_iou: float,
) -> tuple[list[bool], int]:
    ground_truth = {
        image.image_id: tuple(item.bbox_xyxy for item in image.objects)
        for image in images
    }
    matched = {image.image_id: set() for image in images}
    ranked = sorted(
        (
            (detection.confidence, image_id, detection)
            for image_id, detections in predictions.items()
            for detection in detections
            if detection.confidence >= minimum_confidence
        ),
        key=lambda item: (-item[0], item[1], item[2].bbox_xyxy),
    )
    outcomes: list[bool] = []
    for _, image_id, detection in ranked:
        candidates = [
            (_iou(detection.bbox_xyxy, box), index)
            for index, box in enumerate(ground_truth[image_id])
            if index not in matched[image_id]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_match = best_iou >= matching_iou
        if is_match:
            matched[image_id].add(best_index)
        outcomes.append(is_match)
    return outcomes, sum(len(items) for items in ground_truth.values())


def _average_precision(outcomes: Sequence[bool], ground_truth_count: int) -> float | None:
    if ground_truth_count == 0:
        return None
    true_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for rank, is_match in enumerate(outcomes, start=1):
        true_positives += int(is_match)
        precisions.append(true_positives / rank)
        recalls.append(true_positives / ground_truth_count)
    return sum(
        max(
            (precision for precision, recall in zip(precisions, recalls, strict=True) if recall >= threshold),
            default=0.0,
        )
        for threshold in (index / 100 for index in range(101))
    ) / 101


def _validate_thresholds(thresholds: EvaluationThresholds) -> None:
    if not isinstance(thresholds, EvaluationThresholds):
        raise DataValidationError("thresholds must be EvaluationThresholds")
    values = (
        thresholds.confidence_floor,
        thresholds.operating_confidence,
        thresholds.nms_iou,
        thresholds.matching_iou,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise DataValidationError("evaluation thresholds must be finite numbers")
    if not 0 < thresholds.confidence_floor <= thresholds.operating_confidence < 1:
        raise DataValidationError("confidence thresholds are invalid")
    if not 0 < thresholds.nms_iou <= 1 or not 0 < thresholds.matching_iou <= 1:
        raise DataValidationError("IoU thresholds must be in (0, 1]")
    aspect = thresholds.max_symmetric_aspect_ratio
    if aspect is not None and (
        isinstance(aspect, bool)
        or not isinstance(aspect, (int, float))
        or not math.isfinite(aspect)
        or aspect <= 1
    ):
        raise DataValidationError(
            "max_symmetric_aspect_ratio must be finite and greater than 1"
        )


def _group_metrics(
    images: Sequence[EvaluationImage],
    predictions: Mapping[str, Sequence[Detection]],
    thresholds: EvaluationThresholds,
) -> dict[str, int | float | None]:
    group_predictions = {image.image_id: predictions[image.image_id] for image in images}
    outcomes, ground_truth_count = _match(
        images,
        group_predictions,
        thresholds.operating_confidence,
        thresholds.matching_iou,
    )
    true_positive_count = sum(outcomes)
    recall = (
        true_positive_count / ground_truth_count if ground_truth_count else None
    )
    return {
        "sample_count": len(images),
        "ground_truth_count": ground_truth_count,
        "true_positive_count": true_positive_count,
        "recall": recall,
        "miss_rate": None if recall is None else 1 - recall,
    }


def evaluate_detector_predictions(
    images: Sequence[EvaluationImage],
    predictions: Mapping[str, Sequence[Detection]],
    thresholds: EvaluationThresholds,
) -> dict[str, Any]:
    _validate_thresholds(thresholds)
    image_ids: set[str] = set()
    for image in images:
        if not isinstance(image, EvaluationImage):
            raise DataValidationError("images must contain EvaluationImage values")
        if not image.image_id or image.image_id in image_ids:
            raise DataValidationError(f"duplicate or empty image ID: {image.image_id!r}")
        if image.difficulty not in {None, "easy", "medium", "hard"}:
            raise DataValidationError(
                f"{image.image_id} has invalid difficulty: {image.difficulty!r}"
            )
        image_ids.add(image.image_id)
        for index, item in enumerate(image.objects):
            if (
                isinstance(item.category_id, bool)
                or not isinstance(item.category_id, int)
                or item.category_id <= 0
            ):
                raise DataValidationError(
                    f"{image.image_id} object {index} has invalid category_id"
                )
            if item.phase not in {"base", "incremental"}:
                raise DataValidationError(
                    f"{image.image_id} object {index} has invalid phase"
                )
            _validate_box(item.bbox_xyxy, f"{image.image_id} object {index}")
    if set(predictions) != image_ids:
        raise DataValidationError("prediction image IDs must exactly match evaluation images")
    normalized: dict[str, tuple[Detection, ...]] = {}
    for image_id, detections in predictions.items():
        normalized_detections: list[Detection] = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                raise DataValidationError(f"{image_id} detection {index} has invalid type")
            _validate_box(detection.bbox_xyxy, f"{image_id} detection {index}")
            if detection.class_index != 0:
                raise DataValidationError(f"{image_id} detection {index} must use class 0")
            if (
                isinstance(detection.confidence, bool)
                or not isinstance(detection.confidence, (int, float))
                or not math.isfinite(detection.confidence)
                or not 0 <= detection.confidence <= 1
            ):
                raise DataValidationError(f"{image_id} detection {index} confidence is invalid")
            normalized_detections.append(detection)
        normalized[image_id] = tuple(normalized_detections)

    ap_outcomes, ground_truth_count = _match(
        images, normalized, thresholds.confidence_floor, thresholds.matching_iou
    )
    operating_outcomes, _ = _match(
        images, normalized, thresholds.operating_confidence, thresholds.matching_iou
    )
    true_positive_count = sum(operating_outcomes)
    false_positive_count = len(operating_outcomes) - true_positive_count
    recall = (
        true_positive_count / ground_truth_count if ground_truth_count else None
    )
    difficulty_metrics = {
        difficulty: _group_metrics(
            tuple(image for image in images if image.difficulty == difficulty),
            normalized,
            thresholds,
        )
        for difficulty in ("easy", "medium", "hard")
    }
    phase_metrics: dict[str, dict[str, int | float | None]] = {}
    for phase in ("base", "incremental"):
        phase_images = tuple(
            EvaluationImage(
                image_id=image.image_id,
                difficulty=image.difficulty,
                objects=tuple(item for item in image.objects if item.phase == phase),
            )
            for image in images
            if any(item.phase == phase for item in image.objects)
        )
        phase_metrics[phase] = _group_metrics(phase_images, normalized, thresholds)
    return {
        "global": {
            "image_count": len(images),
            "ground_truth_count": ground_truth_count,
            "prediction_count": len(operating_outcomes),
            "true_positive_count": true_positive_count,
            "false_positive_count": false_positive_count,
            "ap50": _average_precision(ap_outcomes, ground_truth_count),
            "recall": recall,
            "miss_rate": None if recall is None else 1 - recall,
        },
        "difficulty": difficulty_metrics,
        "phase": phase_metrics,
    }
