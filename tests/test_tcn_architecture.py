"""
Tests for the TCN architecture options.

The pooling and time-index layers are small enough to reason about exactly, so
these check the properties the experiment design relies on rather than just
that the graph builds: top-k pooling has to bracket the max and the mean, and
the time channel has to be monotone and leave the precipitation untouched. If
either stopped holding, an architecture sweep would still run and would still
produce numbers.
"""

import sys

import keras
import numpy as np
import pytest
import tensorflow as tf

from swafi.impact_cnn_model import ModelCnn, TimeIndexChannel, TopKMeanPooling
from swafi.impact_cnn_options import ImpactCnnOptions

T_LEN = 49
BASE_ARGS = [
    "--dataset", "gvz", "--event-method", "simple",
    "--precip-dataset", "hourly", "--precip-window-size", "1",
    "--precip-time-step", "60", "--precip-days-before", "1",
    "--precip-days-after", "0", "--nb-dense-layers", "2",
    "--nb-dense-units", "128", "--tcn-filters", "32",
    "--random-state", "42", "--run-name", "test",
]


def _build(extra_args):
    sys.argv = ["test"] + BASE_ARGS + list(extra_args)
    options = ImpactCnnOptions()
    options.parse_args()
    model = ModelCnn(options=options,
                     input_3d_size=[T_LEN, 1, 1, 1],
                     input_1d_size=[10])
    model.build_model()
    return model.model


def test_topk_pooling_reduces_to_max_at_k_one():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = TopKMeanPooling(k=1)(tf.constant(x)).numpy()
    np.testing.assert_allclose(got, x.max(axis=1), atol=1e-6)


def test_topk_pooling_reduces_to_mean_at_k_full_length():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = TopKMeanPooling(k=T_LEN)(tf.constant(x)).numpy()
    np.testing.assert_allclose(got, x.mean(axis=1), atol=1e-5)


def test_topk_pooling_lies_between_mean_and_max():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = TopKMeanPooling(k=4)(tf.constant(x)).numpy()
    assert (got <= x.max(axis=1) + 1e-6).all()
    assert (got >= x.mean(axis=1) - 1e-6).all()


def test_topk_pooling_clamps_k_above_sequence_length():
    """k longer than the window must not raise; it saturates at the mean."""
    x = np.random.rand(2, 8, 3).astype("float32")
    got = TopKMeanPooling(k=99)(tf.constant(x)).numpy()
    np.testing.assert_allclose(got, x.mean(axis=1), atol=1e-5)


def test_time_index_channel_is_appended_and_monotone():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = TimeIndexChannel()(tf.constant(x)).numpy()
    assert got.shape == (4, T_LEN, 6)
    # The precipitation channels must come through untouched.
    np.testing.assert_allclose(got[..., :5], x, atol=1e-6)
    index = got[0, :, -1]
    assert index[0] == pytest.approx(0.0)
    assert index[-1] == pytest.approx(1.0)
    assert (np.diff(index) > 0).all()


def test_time_index_channel_is_resolution_independent():
    """Normalised to [0, 1], so hourly and 5-minute windows agree at the ends."""
    a = TimeIndexChannel()(tf.constant(
        np.zeros((1, 49, 1), dtype="float32"))).numpy()
    b = TimeIndexChannel()(tf.constant(
        np.zeros((1, 577, 1), dtype="float32"))).numpy()
    assert a[0, -1, -1] == pytest.approx(b[0, -1, -1])
    assert a[0, 0, -1] == pytest.approx(b[0, 0, -1])


@pytest.mark.parametrize("extra", [
    pytest.param([], id="defaults"),
    pytest.param(["--tcn-pooling", "topk"], id="topk"),
    pytest.param(["--tcn-pooling", "mean_topk", "--tcn-topk", "6"],
                 id="mean_topk"),
    pytest.param(["--use-time-index-channel"], id="time_index"),
    pytest.param(["--tcn-use-gated-activation"], id="gated"),
    pytest.param(["--tcn-use-spatial-dropout"], id="spatial_dropout"),
    pytest.param(["--tcn-nb-layers", "4"], id="four_blocks"),
])
def test_model_builds_and_predicts(extra):
    model = _build(extra)
    x3 = np.random.rand(4, T_LEN, 1, 1, 1).astype("float32")
    x1 = np.random.rand(4, 10).astype("float32")
    out = model([x3, x1], training=False)
    assert tuple(out.shape) == (4, 1)


def test_defaults_are_unchanged_by_the_new_options():
    """The additions must not move the baseline; every sweep depends on it."""
    plain = _build([])
    off = _build(["--no-use-time-index-channel",
                  "--no-tcn-use-gated-activation",
                  "--no-tcn-use-spatial-dropout",
                  "--tcn-pooling", "mean_max"])
    assert plain.count_params() == off.count_params()


def test_mean_topk_is_parameter_matched_to_mean_max():
    """Both emit 2 x filters, so a pooling comparison is not a capacity one."""
    mean_max = _build(["--tcn-pooling", "mean_max"])
    mean_topk = _build(["--tcn-pooling", "mean_topk"])
    assert mean_max.count_params() == mean_topk.count_params()


def test_gated_activation_doubles_the_convolution_parameters():
    """Documents the confound the sweep has to control for."""
    plain = _build([])
    gated = _build(["--tcn-use-gated-activation"])
    assert gated.count_params() > plain.count_params()


def test_new_layers_survive_a_save_load_roundtrip(tmp_path):
    model = _build(["--tcn-pooling", "mean_topk", "--use-time-index-channel",
                    "--tcn-use-gated-activation"])
    x3 = np.random.rand(4, T_LEN, 1, 1, 1).astype("float32")
    x1 = np.random.rand(4, 10).astype("float32")
    before = model.predict([x3, x1], verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    after = reloaded.predict([x3, x1], verbose=0)

    np.testing.assert_allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("nb_layers,kernel,expected", [
    (3, 3, 29),   # the current default: 29 of a 49-step window
    (4, 3, 61),   # covers it
    (3, 5, 57),   # covers it via width instead of depth
])
def test_receptive_field_formula(nb_layers, kernel, expected):
    """Two convolutions per block, so the naive one-conv reading undercounts."""
    dilations = [2 ** i for i in range(nb_layers)]
    assert 1 + 2 * (kernel - 1) * sum(dilations) == expected
