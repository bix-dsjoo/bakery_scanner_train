import pytest

from bakery_scanner.detector_evaluation import (
    Detection,
    EvaluationImage,
    EvaluationObject,
    EvaluationThresholds,
    evaluate_detector_predictions,
)


def test_evaluate_detector_predictions_computes_global_metrics() -> None:
    images = (
        EvaluationImage(
            "scene_e_1.jpg",
            "easy",
            (EvaluationObject((0, 0, 10, 10), 2, "base"),),
        ),
        EvaluationImage(
            "scene_h_1.jpg",
            "hard",
            (EvaluationObject((20, 20, 30, 30), 4, "base"),),
        ),
    )
    predictions = {
        "scene_e_1.jpg": (
            Detection((0, 0, 10, 10), 0.9),
            Detection((0, 0, 10, 10), 0.8),
        ),
        "scene_h_1.jpg": (),
    }

    metrics = evaluate_detector_predictions(
        images, predictions, EvaluationThresholds()
    )

    assert metrics["global"]["ground_truth_count"] == 2
    assert metrics["global"]["true_positive_count"] == 1
    assert metrics["global"]["false_positive_count"] == 1
    assert metrics["global"]["recall"] == pytest.approx(0.5)
    assert metrics["global"]["miss_rate"] == pytest.approx(0.5)
    assert metrics["global"]["ap50"] == pytest.approx(51 / 101)


def test_phase_without_objects_uses_null_metrics() -> None:
    images = (
        EvaluationImage(
            "scene_m_1.jpg",
            "medium",
            (EvaluationObject((1, 1, 5, 5), 2, "base"),),
        ),
    )

    metrics = evaluate_detector_predictions(
        images, {"scene_m_1.jpg": ()}, EvaluationThresholds()
    )

    assert metrics["phase"]["incremental"] == {
        "sample_count": 0,
        "ground_truth_count": 0,
        "true_positive_count": 0,
        "recall": None,
        "miss_rate": None,
    }
    assert metrics["phase"]["base"]["sample_count"] == 1
    assert metrics["phase"]["base"]["ground_truth_count"] == 1
    assert metrics["phase"]["base"]["recall"] == pytest.approx(0.0)


def test_difficulty_groups_include_empty_and_missing_groups() -> None:
    images = (
        EvaluationImage(
            "scene_m_1.jpg",
            "medium",
            (EvaluationObject((1, 1, 5, 5), 2, "base"),),
        ),
        EvaluationImage("scene_e_2.jpg", "easy", ()),
    )

    metrics = evaluate_detector_predictions(
        images,
        {"scene_m_1.jpg": (), "scene_e_2.jpg": ()},
        EvaluationThresholds(),
    )

    assert metrics["difficulty"]["medium"]["sample_count"] == 1
    assert metrics["difficulty"]["medium"]["recall"] == pytest.approx(0.0)
    assert metrics["difficulty"]["easy"]["sample_count"] == 1
    assert metrics["difficulty"]["easy"]["ground_truth_count"] == 0
    assert metrics["difficulty"]["easy"]["recall"] is None
    assert metrics["difficulty"]["hard"]["sample_count"] == 0
    assert metrics["difficulty"]["hard"]["recall"] is None
