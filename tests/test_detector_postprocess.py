import pytest

from bakery_scanner.detector_evaluation import Detection
from bakery_scanner.detector_postprocess import DetectorPostprocessConfig, filter_detections
from bakery_scanner.errors import DataValidationError


def test_filter_detections_removes_overlap_and_extreme_aspect_ratio() -> None:
    detections = (
        Detection((0.0, 0.0, 100.0, 100.0), 0.90),
        Detection((5.0, 5.0, 105.0, 105.0), 0.80),
        Detection((130.0, 0.0, 230.0, 100.0), 0.70),
        Detection((250.0, 0.0, 270.0, 100.0), 0.85),
    )

    filtered = filter_detections(
        detections,
        DetectorPostprocessConfig(
            confidence_threshold=0.05,
            nms_iou=0.15,
            max_symmetric_aspect_ratio=2.0,
        ),
    )

    assert filtered == (detections[0], detections[2])


def test_filter_detections_applies_confidence_before_nms() -> None:
    low_overlap = Detection((0.0, 0.0, 100.0, 100.0), 0.04)
    eligible = Detection((5.0, 5.0, 105.0, 105.0), 0.06)

    filtered = filter_detections(
        (low_overlap, eligible),
        DetectorPostprocessConfig(0.05, 0.15, 2.0),
    )

    assert filtered == (eligible,)


@pytest.mark.parametrize(
    "config",
    [
        DetectorPostprocessConfig(0.0, 0.15, 2.0),
        DetectorPostprocessConfig(1.0, 0.15, 2.0),
        DetectorPostprocessConfig(0.05, 0.0, 2.0),
        DetectorPostprocessConfig(0.05, 1.1, 2.0),
        DetectorPostprocessConfig(0.05, 0.15, 1.0),
    ],
)
def test_filter_detections_rejects_invalid_config(
    config: DetectorPostprocessConfig,
) -> None:
    with pytest.raises(DataValidationError):
        filter_detections((), config)
