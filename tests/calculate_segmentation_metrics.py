import numpy as np

from image_analysis.metrics import calculate_metrics

def test_calculate_segmentation_metrics_perfect_match():
    ground_truth = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    prediction = ground_truth.copy()

    metrics = calculate_segmentation_metrics(prediction, ground_truth)

    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

