from __future__ import annotations

import math

import pytest

from bakery_scanner.classifier_evaluation import (
    ClassifierPrediction,
    evaluate_classifier_predictions,
)
from bakery_scanner.errors import DataValidationError


def test_classifier_metrics_use_supported_classes_for_macro_f1() -> None:
    predictions = [
        ClassifierPrediction("a", 0, 0, 0.9),
        ClassifierPrediction("b", 1, 1, 0.8),
        ClassifierPrediction("c", 1, 0, 0.7),
        ClassifierPrediction("d", 2, 2, 0.95),
    ]

    metrics = evaluate_classifier_predictions(predictions, output_dimension=4)

    assert metrics["sample_count"] == 4
    assert metrics["evaluated_class_count"] == 3
    assert metrics["top1_accuracy"] == 0.75
    assert math.isclose(metrics["macro_f1"], 7 / 9)
    assert metrics["per_class"]["0"] == {
        "support": 1,
        "predicted_count": 2,
        "true_positive_count": 1,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
    }
    assert metrics["per_class"]["1"]["precision"] == 1.0
    assert metrics["per_class"]["1"]["recall"] == 0.5
    assert metrics["per_class"]["2"]["f1"] == 1.0
    assert metrics["per_class"]["3"] == {
        "support": 0,
        "predicted_count": 0,
        "true_positive_count": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def test_absent_class_with_false_positive_has_zero_precision() -> None:
    metrics = evaluate_classifier_predictions(
        [ClassifierPrediction("a", 0, 1, 0.6)], output_dimension=2
    )
    assert metrics["per_class"]["1"]["precision"] == 0.0
    assert metrics["per_class"]["1"]["recall"] is None
    assert metrics["per_class"]["1"]["f1"] is None


def test_incremental_metrics_separate_base_and_new_class_performance() -> None:
    predictions = [
        ClassifierPrediction("base-correct", 0, 0, 0.9),
        ClassifierPrediction("base-wrong", 1, 0, 0.8),
        ClassifierPrediction("new-correct", 15, 15, 0.95),
        ClassifierPrediction("new-wrong", 16, 15, 0.7),
    ]

    metrics = evaluate_classifier_predictions(predictions, output_dimension=20)

    assert metrics["phase"]["base"]["sample_count"] == 2
    assert metrics["phase"]["base"]["top1_accuracy"] == 0.5
    assert math.isclose(metrics["phase"]["base"]["macro_f1"], 1 / 3)
    assert metrics["phase"]["incremental"]["sample_count"] == 2
    assert metrics["phase"]["incremental"]["top1_accuracy"] == 0.5
    assert math.isclose(
        metrics["phase"]["incremental"]["macro_f1"], 1 / 3
    )


@pytest.mark.parametrize(
    "predictions,dimension,message",
    [
        ([], 2, "at least one"),
        ([ClassifierPrediction("a", 0, 0, 0.5)], 0, "output_dimension"),
        ([ClassifierPrediction("a", 2, 0, 0.5)], 2, "target"),
        ([ClassifierPrediction("a", 0, 2, 0.5)], 2, "predicted"),
        ([ClassifierPrediction("a", 0, 0, 1.1)], 2, "confidence"),
        (
            [
                ClassifierPrediction("a", 0, 0, 0.5),
                ClassifierPrediction("a", 1, 1, 0.5),
            ],
            2,
            "duplicate",
        ),
    ],
)
def test_classifier_metrics_reject_invalid_predictions(
    predictions,
    dimension: int,
    message: str,
) -> None:
    with pytest.raises(DataValidationError, match=message):
        evaluate_classifier_predictions(predictions, output_dimension=dimension)
