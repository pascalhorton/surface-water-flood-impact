"""
Tests for the NumPy MiniRocket implementation.
"""
import numpy as np
import pytest

from swafi.utils.minirocket import MiniRocket, NUM_KERNELS


def test_minirocket_output_shape_and_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 97)).astype(np.float32)

    rocket = MiniRocket(num_features=9996, random_state=0)
    feats = rocket.fit(x).transform(x)

    assert feats.shape == (32, rocket.num_features)
    assert rocket.num_features % NUM_KERNELS == 0
    assert feats.dtype == np.float32
    assert feats.min() >= 0.0
    assert feats.max() <= 1.0
    assert np.isfinite(feats).all()


def test_minirocket_deterministic_with_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 60)).astype(np.float32)

    feats_a = MiniRocket(random_state=42).fit(x).transform(x)
    feats_b = MiniRocket(random_state=42).fit(x).transform(x)

    np.testing.assert_array_equal(feats_a, feats_b)


def test_minirocket_chunking_consistent():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 50)).astype(np.float32)

    rocket = MiniRocket(random_state=0).fit(x)
    np.testing.assert_allclose(
        rocket.transform(x, chunk_size=7),
        rocket.transform(x, chunk_size=100))


def test_minirocket_rejects_wrong_length():
    rng = np.random.default_rng(3)
    rocket = MiniRocket(random_state=0)
    rocket.fit(rng.normal(size=(10, 50)))
    with pytest.raises(AssertionError):
        rocket.transform(rng.normal(size=(10, 40)))


def test_minirocket_features_are_discriminative():
    # Class 0: white noise; class 1: noise plus short spikes. A linear
    # classifier on the MiniRocket features must separate them easily.
    from sklearn.linear_model import RidgeClassifierCV

    rng = np.random.default_rng(4)
    n_per_class, t_len = 60, 97
    x0 = rng.normal(size=(n_per_class, t_len))
    x1 = rng.normal(size=(n_per_class, t_len))
    spike_pos = rng.integers(10, t_len - 10, size=(n_per_class, 3))
    for i in range(n_per_class):
        x1[i, spike_pos[i]] += 6.0

    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.repeat([0, 1], n_per_class)
    shuffled = rng.permutation(len(y))
    x, y = x[shuffled], y[shuffled]

    rocket = MiniRocket(num_features=2520, random_state=0)
    feats_train = rocket.fit(x[:80]).transform(x[:80])
    feats_test = rocket.transform(x[80:])

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(feats_train, y[:80])
    accuracy = clf.score(feats_test, y[80:])
    assert accuracy >= 0.9, f"accuracy too low: {accuracy}"
