from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .errors import DataValidationError


@dataclass(frozen=True, slots=True)
class ClassifierPrediction:
    sample_id: str
    target_index: int
    predicted_index: int
    confidence: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "sample_id": self.sample_id,
            "target_index": self.target_index,
            "predicted_index": self.predicted_index,
            "confidence": self.confidence,
        }


def evaluate_classifier_predictions(
    predictions: Sequence[ClassifierPrediction],
    *,
    output_dimension: int,
) -> dict[str, Any]:
    """Compute JSON-safe classifier metrics without relying on test-only tooling."""

    if (
        isinstance(output_dimension, bool)
        or not isinstance(output_dimension, int)
        or output_dimension <= 0
    ):
        raise DataValidationError("output_dimension must be a positive integer")
    if not predictions:
        raise DataValidationError("predictions must contain at least one sample")

    sample_ids: set[str] = set()
    support = [0] * output_dimension
    predicted_count = [0] * output_dimension
    true_positive_count = [0] * output_dimension

    for prediction in predictions:
        if not isinstance(prediction, ClassifierPrediction):
            raise DataValidationError(
                "predictions must contain ClassifierPrediction instances"
            )
        if not isinstance(prediction.sample_id, str) or not prediction.sample_id:
            raise DataValidationError("sample_id must be a non-empty string")
        if prediction.sample_id in sample_ids:
            raise DataValidationError(
                f"duplicate sample_id in predictions: {prediction.sample_id}"
            )
        sample_ids.add(prediction.sample_id)

        _validate_index("target_index", prediction.target_index, output_dimension)
        _validate_index("predicted_index", prediction.predicted_index, output_dimension)
        if (
            isinstance(prediction.confidence, bool)
            or not isinstance(prediction.confidence, (int, float))
            or not math.isfinite(prediction.confidence)
            or not 0 <= prediction.confidence <= 1
        ):
            raise DataValidationError("confidence must be a finite number between 0 and 1")

        target = prediction.target_index
        predicted = prediction.predicted_index
        support[target] += 1
        predicted_count[predicted] += 1
        if target == predicted:
            true_positive_count[target] += 1

    per_class: dict[str, dict[str, int | float | None]] = {}
    supported_f1: list[float] = []
    for index in range(output_dimension):
        class_support = support[index]
        class_predictions = predicted_count[index]
        true_positives = true_positive_count[index]
        precision = (
            true_positives / class_predictions if class_predictions > 0 else None
        )
        recall = true_positives / class_support if class_support > 0 else None
        if class_support == 0:
            f1 = None
        elif true_positives == 0:
            f1 = 0.0
        else:
            assert precision is not None and recall is not None
            f1 = 2 * precision * recall / (precision + recall)
        if f1 is not None:
            supported_f1.append(f1)

        per_class[str(index)] = {
            "support": class_support,
            "predicted_count": class_predictions,
            "true_positive_count": true_positives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_correct = sum(true_positive_count)
    metrics = {
        "sample_count": len(predictions),
        "evaluated_class_count": len(supported_f1),
        "top1_accuracy": total_correct / len(predictions),
        "macro_f1": sum(supported_f1) / len(supported_f1),
        "per_class": per_class,
    }
    if output_dimension in {15, 20}:
        groups = {"base": range(15)}
        if output_dimension == 20:
            groups["incremental"] = range(15, 20)
        phase_metrics: dict[str, dict[str, int | float | None]] = {}
        for name, indices in groups.items():
            index_set = set(indices)
            group_predictions = [
                prediction
                for prediction in predictions
                if prediction.target_index in index_set
            ]
            f1_values = [
                per_class[str(index)]["f1"]
                for index in index_set
                if per_class[str(index)]["f1"] is not None
            ]
            phase_metrics[name] = {
                "sample_count": len(group_predictions),
                "top1_accuracy": (
                    sum(
                        prediction.target_index == prediction.predicted_index
                        for prediction in group_predictions
                    )
                    / len(group_predictions)
                    if group_predictions
                    else None
                ),
                "macro_f1": (
                    sum(float(value) for value in f1_values) / len(f1_values)
                    if f1_values
                    else None
                ),
            }
        metrics["phase"] = phase_metrics
    return metrics


def _validate_index(label: str, value: int, output_dimension: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < output_dimension
    ):
        raise DataValidationError(
            f"{label} must be an integer between 0 and {output_dimension - 1}"
        )
