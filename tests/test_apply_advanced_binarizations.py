import numpy as np

from image_analysis.pipeline import apply_advanced_binarizations


def test_apply_advanced_binarizations_returns_binary_masks():
    image = np.array(
        [
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    results = apply_advanced_binarizations(image)

    assert isinstance(results, dict)
    assert len(results) > 0

    for method_name, mask in results.items():
        assert isinstance(method_name, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == image.shape

        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, 1, 255, False, True})
