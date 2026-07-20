from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import DataValidationError

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True, slots=True)
class EndToEndTruth:
    bbox_xyxy: tuple[float, float, float, float]
    model_index: int


@dataclass(frozen=True, slots=True)
class EndToEndImage:
    image_id: str
    objects: tuple[EndToEndTruth, ...]


@dataclass(frozen=True, slots=True)
class EndToEndPrediction:
    image_id: str
    bbox_xyxy: tuple[float, float, float, float]
    model_index: int
    detector_confidence: float
    classifier_confidence: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "model_index": self.model_index,
            "detector_confidence": self.detector_confidence,
            "classifier_confidence": self.classifier_confidence,
            "score": self.score,
        }


def _box(
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


def _index(value: int, output_dimension: int, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < output_dimension
    ):
        raise DataValidationError(
            f"{label} model_index must be between 0 and {output_dimension - 1}"
        )
    return value


def _confidence(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise DataValidationError(f"{label} must be a finite number between 0 and 1")
    return float(value)


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


def _average_precision(outcomes: Sequence[bool], ground_truth_count: int) -> float:
    true_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for rank, outcome in enumerate(outcomes, start=1):
        true_positives += int(outcome)
        precisions.append(true_positives / rank)
        recalls.append(true_positives / ground_truth_count)
    return sum(
        max(
            (
                precision
                for precision, recall in zip(precisions, recalls, strict=True)
                if recall >= threshold
            ),
            default=0.0,
        )
        for threshold in (index / 100 for index in range(101))
    ) / 101


def _class_ap(
    images: Sequence[EndToEndImage],
    predictions: Mapping[str, Sequence[EndToEndPrediction]],
    model_index: int,
    iou_threshold: float,
) -> tuple[float | None, int, int]:
    ground_truth = {
        image.image_id: tuple(
            item.bbox_xyxy
            for item in image.objects
            if item.model_index == model_index
        )
        for image in images
    }
    ground_truth_count = sum(len(items) for items in ground_truth.values())
    ranked = sorted(
        (
            prediction
            for items in predictions.values()
            for prediction in items
            if prediction.model_index == model_index
        ),
        key=lambda item: (-item.score, item.image_id, item.bbox_xyxy),
    )
    if ground_truth_count == 0:
        return None, 0, len(ranked)
    matched = {image_id: set() for image_id in ground_truth}
    outcomes: list[bool] = []
    for prediction in ranked:
        candidates = [
            (_iou(prediction.bbox_xyxy, truth_box), position)
            for position, truth_box in enumerate(ground_truth[prediction.image_id])
            if position not in matched[prediction.image_id]
        ]
        best_iou, best_position = max(candidates, default=(0.0, -1))
        is_match = best_iou >= iou_threshold
        if is_match:
            matched[prediction.image_id].add(best_position)
        outcomes.append(is_match)
    return _average_precision(outcomes, ground_truth_count), ground_truth_count, len(ranked)


def evaluate_end_to_end_predictions(
    images: Sequence[EndToEndImage],
    predictions: Mapping[str, Sequence[EndToEndPrediction]],
    *,
    output_dimension: int,
    count_detector_confidence: float = 0.0,
) -> dict[str, Any]:
    if (
        isinstance(output_dimension, bool)
        or not isinstance(output_dimension, int)
        or output_dimension <= 0
    ):
        raise DataValidationError("output_dimension must be a positive integer")
    if not images:
        raise DataValidationError("images must contain at least one evaluation image")
    count_threshold = _confidence(
        count_detector_confidence, "count detector confidence"
    )

    image_ids: set[str] = set()
    normalized_images: list[EndToEndImage] = []
    for image in images:
        if not isinstance(image, EndToEndImage):
            raise DataValidationError("images must contain EndToEndImage values")
        if not isinstance(image.image_id, str) or not image.image_id:
            raise DataValidationError("image_id must be a non-empty string")
        if image.image_id in image_ids:
            raise DataValidationError(f"duplicate image_id: {image.image_id}")
        image_ids.add(image.image_id)
        objects = []
        for position, item in enumerate(image.objects):
            if not isinstance(item, EndToEndTruth):
                raise DataValidationError(
                    f"{image.image_id} truth {position} has invalid type"
                )
            objects.append(
                EndToEndTruth(
                    _box(item.bbox_xyxy, f"{image.image_id} truth {position}"),
                    _index(
                        item.model_index,
                        output_dimension,
                        f"{image.image_id} truth {position}",
                    ),
                )
            )
        normalized_images.append(EndToEndImage(image.image_id, tuple(objects)))

    if set(predictions) != image_ids:
        raise DataValidationError("prediction image IDs must exactly match evaluation image IDs")
    normalized_predictions: dict[str, tuple[EndToEndPrediction, ...]] = {}
    for image_id, items in predictions.items():
        normalized = []
        for position, item in enumerate(items):
            if not isinstance(item, EndToEndPrediction):
                raise DataValidationError(
                    f"{image_id} prediction {position} has invalid type"
                )
            if item.image_id != image_id:
                raise DataValidationError(
                    f"{image_id} prediction {position} image_id does not match"
                )
            detector_confidence = _confidence(
                item.detector_confidence,
                f"{image_id} prediction {position} detector confidence",
            )
            classifier_confidence = _confidence(
                item.classifier_confidence,
                f"{image_id} prediction {position} classifier confidence",
            )
            score = _confidence(item.score, f"{image_id} prediction {position} score")
            if not math.isclose(
                score,
                detector_confidence * classifier_confidence,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise DataValidationError(
                    f"{image_id} prediction {position} score must equal confidence product"
                )
            normalized.append(
                EndToEndPrediction(
                    image_id,
                    _box(item.bbox_xyxy, f"{image_id} prediction {position}"),
                    _index(
                        item.model_index,
                        output_dimension,
                        f"{image_id} prediction {position}",
                    ),
                    detector_confidence,
                    classifier_confidence,
                    score,
                )
            )
        normalized_predictions[image_id] = tuple(normalized)
    operating_predictions = {
        image_id: tuple(
            item
            for item in items
            if item.detector_confidence >= count_threshold
        )
        for image_id, items in normalized_predictions.items()
    }

    class_metrics: dict[str, dict[str, int | float | None]] = {}
    supported_ap50: list[float] = []
    supported_all_aps: list[float] = []
    all_count_accuracies: list[float] = []
    supported_count_accuracies: list[float] = []
    for model_index in range(output_dimension):
        aps: list[float] = []
        ground_truth_count = 0
        prediction_count = 0
        for threshold in IOU_THRESHOLDS:
            ap, ground_truth_count, prediction_count = _class_ap(
                normalized_images,
                normalized_predictions,
                model_index,
                threshold,
            )
            if ap is not None:
                aps.append(ap)
        ap50 = aps[0] if aps else None
        ap50_95 = sum(aps) / len(aps) if aps else None
        exact_images = sum(
            sum(item.model_index == model_index for item in image.objects)
            == sum(
                item.model_index == model_index
                for item in operating_predictions[image.image_id]
            )
            for image in normalized_images
        )
        operating_prediction_count = sum(
            item.model_index == model_index
            for items in operating_predictions.values()
            for item in items
        )
        exact_count_accuracy = exact_images / len(normalized_images)
        all_count_accuracies.append(exact_count_accuracy)
        if ap50 is not None:
            supported_ap50.append(ap50)
            supported_all_aps.extend(aps)
            supported_count_accuracies.append(exact_count_accuracy)
        class_metrics[str(model_index)] = {
            "ground_truth_count": ground_truth_count,
            "prediction_count": prediction_count,
            "operating_prediction_count": operating_prediction_count,
            "ap50": ap50,
            "ap50_95": ap50_95,
            "exact_count_accuracy": exact_count_accuracy,
        }

    return {
        "image_count": len(normalized_images),
        "ground_truth_count": sum(
            len(image.objects) for image in normalized_images
        ),
        "prediction_count": sum(len(items) for items in normalized_predictions.values()),
        "operating_prediction_count": sum(
            len(items) for items in operating_predictions.values()
        ),
        "count_detector_confidence": count_threshold,
        "evaluated_class_count": len(supported_ap50),
        "map50": (
            sum(supported_ap50) / len(supported_ap50)
            if supported_ap50
            else None
        ),
        "map50_95": (
            sum(supported_all_aps) / len(supported_all_aps)
            if supported_all_aps
            else None
        ),
        "macro_exact_count_accuracy": sum(all_count_accuracies)
        / len(all_count_accuracies),
        "supported_macro_exact_count_accuracy": (
            sum(supported_count_accuracies) / len(supported_count_accuracies)
            if supported_count_accuracies
            else None
        ),
        "per_class": class_metrics,
    }
