import numpy as np

from my_cil.data.patches import make_patches


def test_make_patches_extracts_non_background_pixels():
    features = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    labels = np.array([[0, 1, 0], [0, 0, 0], [2, 0, 0]])
    patches, patch_labels, index = make_patches(features, labels, window_size=3)
    assert patches.shape == (2, 3, 3, 2)
    np.testing.assert_array_equal(patch_labels, [1, 2])
    np.testing.assert_array_equal(index, [[1, 0, 1], [2, 2, 0]])
