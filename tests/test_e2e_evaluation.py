from __future__ import annotations

import math

import pytest

from bakery_scanner.e2e_evaluation import (
    EndToEndImage,
    EndToEndPrediction,
    EndToEndTruth,
    evaluate_end_to_end_predictions,
)
from bakery_scanner.errors import DataValidationError


def _prediction(
    image_id: str,
    bbox: tuple[float, float, float, float],
    model_index: int,
    detector: float,
    classifier: float,
) -> EndToEndPrediction:
    return EndToEndPrediction(
        image_id=image_id,
        bbox_xyxy=bbox,
        model_index=model_index,
        detector_confidence=detector,
        classifier_confidence=classifier,
        score=detector * classifier,
    )


def test_e2e_metrics_are_class_aware_and_record_exact_counts() -> None:
    images = (
        EndToEndImage(
            "one",
            (
                EndToEndTruth((0.0, 0.0, 10.0, 10.0), 0),
                EndToEndTruth((20.0, 0.0, 30.0, 10.0), 1),
            ),
        ),
        EndToEndImage(
            "two", (EndToEndTruth((0.0, 0.0, 10.0, 10.0), 0),)
        ),
    )
    predictions = {
        "one": (
            _prediction("one", (0.0, 0.0, 10.0, 10.0), 0, 0.9, 0.9),
            _prediction("one", (20.0, 0.0, 30.0, 10.0), 2, 0.8, 0.8),
        ),
        "two": (
            _prediction("two", (0.0, 0.0, 10.0, 10.0), 0, 0.7, 0.9),
        ),
    }

    metrics = evaluate_end_to_end_predictions(
        images, predictions, output_dimension=3
    )

    assert metrics["image_count"] == 2
    assert metrics["ground_truth_count"] == 3
    assert metrics["prediction_count"] == 3
    assert metrics["evaluated_class_count"] == 2
    assert metrics["map50"] == 0.5
    assert metrics["map50_95"] == 0.5
    assert metrics["per_class"]["0"]["ap50"] == 1.0
    assert metrics["per_class"]["1"]["ap50"] == 0.0
    assert metrics["per_class"]["2"]["ap50"] is None
    assert metrics["per_class"]["0"]["exact_count_accuracy"] == 1.0
    assert metrics["per_class"]["1"]["exact_count_accuracy"] == 0.5
    assert metrics["per_class"]["2"]["exact_count_accuracy"] == 0.5
    assert math.isclose(metrics["macro_exact_count_accuracy"], 2 / 3)
    assert metrics["supported_macro_exact_count_accuracy"] == 0.75


def test_higher_ranked_false_positive_reduces_average_precision() -> None:
    images = (
        EndToEndImage(
            "one", (EndToEndTruth((0.0, 0.0, 10.0, 10.0), 0),)
        ),
    )
    predictions = {
        "one": (
            _prediction("one", (20.0, 20.0, 30.0, 30.0), 0, 1.0, 0.9),
            _prediction("one", (0.0, 0.0, 10.0, 10.0), 0, 0.8, 1.0),
        )
    }

    metrics = evaluate_end_to_end_predictions(
        images, predictions, output_dimension=1
    )

    assert metrics["map50"] == 0.5
    assert metrics["map50_95"] == 0.5


def test_ap_uses_floor_predictions_but_counts_use_operating_threshold() -> None:
    images = (
        EndToEndImage(
            "one", (EndToEndTruth((0.0, 0.0, 10.0, 10.0), 0),)
        ),
    )
    predictions = {
        "one": (
            _prediction("one", (0.0, 0.0, 10.0, 10.0), 0, 0.1, 0.9),
        )
    }

    metrics = evaluate_end_to_end_predictions(
        images,
        predictions,
        output_dimension=1,
        count_detector_confidence=0.25,
    )

    assert metrics["map50"] == 1.0
    assert metrics["prediction_count"] == 1
    assert metrics["operating_prediction_count"] == 0
    assert metrics["per_class"]["0"]["exact_count_accuracy"] == 0.0


def test_empty_predictions_are_valid_and_score_zero_ap() -> None:
    images = (
        EndToEndImage(
            "one", (EndToEndTruth((0.0, 0.0, 10.0, 10.0), 0),)
        ),
    )

    metrics = evaluate_end_to_end_predictions(
        images, {"one": ()}, output_dimension=2
    )

    assert metrics["prediction_count"] == 0
    assert metrics["map50"] == 0.0
    assert metrics["per_class"]["0"]["exact_count_accuracy"] == 0.0
    assert metrics["per_class"]["1"]["exact_count_accuracy"] == 1.0


@pytest.mark.parametrize(
    "images,predictions,dimension,message",
    [
        ((), {}, 1, "at least one"),
        ((EndToEndImage("one", ()),), {"one": ()}, 0, "output_dimension"),
        (
            (EndToEndImage("one", (EndToEndTruth((0, 0, 0, 1), 0),)),),
            {"one": ()},
            1,
            "positive area",
        ),
        (
            (EndToEndImage("one", (EndToEndTruth((0, 0, 1, 1), 1),)),),
            {"one": ()},
            1,
            "model_index",
        ),
        (
            (EndToEndImage("one", ()),),
            {"other": ()},
            1,
            "image IDs",
        ),
        (
            (EndToEndImage("one", ()),),
            {
                "one": (
                    EndToEndPrediction(
                        "one", (0, 0, 1, 1), 0, 0.5, 0.5, 0.3
                    ),
                )
            },
            1,
            "product",
        ),
    ],
)
def test_e2e_metrics_reject_invalid_records(
    images, predictions, dimension: int, message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        evaluate_end_to_end_predictions(
            images, predictions, output_dimension=dimension
        )
