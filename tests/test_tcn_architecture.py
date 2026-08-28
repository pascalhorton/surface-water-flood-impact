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

from swafi.impact_cnn_model import (ModelCnn, SegmentMaxPooling,
                                    SoftArgmaxPooling, TimeIndexChannel,
                                    TopKMeanPooling)
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
    pytest.param(["--tcn-dilation-base", "3"], id="dilation_base_3"),
    pytest.param(["--tcn-pooling", "segmax"], id="segmax"),
    pytest.param(["--tcn-pooling", "segmax", "--tcn-nb-segments", "4"],
                 id="segmax_4"),
    pytest.param(["--tcn-pooling", "mean_segmax"], id="mean_segmax"),
    pytest.param(["--tcn-pooling", "softargmax"], id="softargmax"),
    pytest.param(["--tcn-pooling", "softargmax", "--tcn-softargmax-beta", "8"],
                 id="softargmax_sharp"),
    pytest.param(["--tcn-pooling", "mean_softargmax"], id="mean_softargmax"),
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


# --- Phase I: dilation base, segment pooling -------------------------------


def test_segment_max_reduces_to_global_max_at_one_segment():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = SegmentMaxPooling(nb_segments=1)(tf.constant(x)).numpy()
    np.testing.assert_allclose(got, x.max(axis=1), atol=1e-6)


def test_segment_max_splits_the_window_and_keeps_order():
    """Two segments must report the early and the late maximum separately."""
    x = np.zeros((1, 10, 1), dtype="float32")
    x[0, 1, 0] = 3.0    # first half
    x[0, 7, 0] = 9.0    # second half
    got = SegmentMaxPooling(nb_segments=2)(tf.constant(x)).numpy()
    assert got.shape == (1, 2)
    np.testing.assert_allclose(got[0], [3.0, 9.0], atol=1e-6)


def test_segment_max_boundaries_follow_the_window_length():
    """A burst 20% into the window lands in segment 0 at either resolution.

    The point of computing boundaries from the runtime length: nb_segments = 2
    has to mean first half / second half hourly and at 5 minutes alike, not a
    fixed number of steps that silently becomes 4% of a sub-hourly window.
    """
    for t_len in (49, 577):
        x = np.zeros((1, t_len, 1), dtype="float32")
        x[0, int(0.2 * t_len), 0] = 5.0
        got = SegmentMaxPooling(nb_segments=2)(tf.constant(x)).numpy()
        assert got[0, 0] == pytest.approx(5.0)
        assert got[0, 1] == pytest.approx(0.0)


def test_segment_max_survives_more_segments_than_steps():
    """Must not emit -inf from an empty slice, which would poison the model."""
    x = np.random.rand(2, 3, 4).astype("float32")
    got = SegmentMaxPooling(nb_segments=8)(tf.constant(x)).numpy()
    assert got.shape == (2, 32)
    assert np.isfinite(got).all()


def test_segmax_two_is_parameter_matched_to_mean_max():
    """Both emit 2 x filters, so the comparison is pooling, not capacity."""
    mean_max = _build(["--tcn-pooling", "mean_max"])
    segmax = _build(["--tcn-pooling", "segmax", "--tcn-nb-segments", "2"])
    assert mean_max.count_params() == segmax.count_params()


def test_more_segments_are_not_parameter_matched():
    """Documents why the sweep uses two segments and not four."""
    two = _build(["--tcn-pooling", "segmax", "--tcn-nb-segments", "2"])
    four = _build(["--tcn-pooling", "segmax", "--tcn-nb-segments", "4"])
    assert four.count_params() > two.count_params()


def test_dilation_base_costs_no_parameters():
    """The whole point of the option: receptive field without capacity."""
    base2 = _build(["--tcn-dilation-base", "2"])
    base3 = _build(["--tcn-dilation-base", "3"])
    assert base2.count_params() == base3.count_params()


@pytest.mark.parametrize("nb_layers,kernel,base,expected", [
    (3, 3, 2, 29),    # the current default, 29 of a 49-step window
    (3, 3, 3, 53),    # covers it, at identical parameter count
    (4, 3, 2, 61),    # covers it by depth, +12% parameters
    (3, 5, 2, 57),    # covers it by kernel width, +23% parameters
])
def test_receptive_field_with_dilation_base(nb_layers, kernel, base, expected):
    dilations = [base ** i for i in range(nb_layers)]
    assert 1 + 2 * (kernel - 1) * sum(dilations) == expected


def test_phase_i_options_do_not_move_the_default():
    """arch_base has to keep reproducing 0.0854; the defaults must not shift."""
    plain = _build([])
    explicit = _build(["--tcn-dilation-base", "2", "--tcn-pooling", "mean_max",
                       "--tcn-nb-segments", "2"])
    assert plain.count_params() == explicit.count_params()


def test_segment_pooling_survives_a_save_load_roundtrip(tmp_path):
    model = _build(["--tcn-pooling", "mean_segmax", "--tcn-nb-segments", "3"])
    x3 = np.random.rand(4, T_LEN, 1, 1, 1).astype("float32")
    x1 = np.random.rand(4, 10).astype("float32")
    before = model.predict([x3, x1], verbose=0)

    path = tmp_path / "segmax.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    # nb_segments has to come back through get_config, or the reloaded model
    # pools differently and the saved weights no longer line up.
    np.testing.assert_allclose(before, reloaded.predict([x3, x1], verbose=0),
                               atol=1e-6)


# --- dense head structure ---------------------------------------------------


def test_dense_head_warns_when_no_skip_is_an_identity(caplog):
    """The default head projects both skips, which is the surprising case.

    Three flags decide it between them and none of them mentions residuals, so
    the warning is the only place a run records that its residual connections
    are parallel linear paths rather than gradient shortcuts.
    """
    with caplog.at_level("INFO", logger="swafi.impact_cnn_model"):
        _build(["--nb-dense-layers", "2", "--nb-dense-units", "128"])
    line = [r for r in caplog.records if "Dense head" in r.getMessage()]
    assert len(line) == 1
    assert line[0].levelname == "WARNING"
    msg = line[0].getMessage()
    assert "0 identity, 2 projected" in msg
    # 74 x 128 + 128 x 64, the pooled width being 64 TCN channels plus the 10
    # tabular inputs of this fixture. Precipitation-only runs read 16,384.
    assert "17,664 parameters" in msg


def test_dense_head_reports_identity_skips_without_warning(caplog):
    """Constant width at the pooled width: every skip is a plain add."""
    with caplog.at_level("INFO", logger="swafi.impact_cnn_model"):
        _build(["--nb-dense-layers", "2", "--nb-dense-units", "74",
                "--no-nb-dense-units-decreasing"])
    line = [r for r in caplog.records if "Dense head" in r.getMessage()]
    assert len(line) == 1
    assert line[0].levelname == "INFO"
    assert "2 identity, 0 projected" in line[0].getMessage()


def test_dense_head_reports_widths_in_order(caplog):
    with caplog.at_level("INFO", logger="swafi.impact_cnn_model"):
        _build(["--nb-dense-layers", "3", "--nb-dense-units", "128"])
    msg = [r.getMessage() for r in caplog.records if "Dense head" in r.getMessage()][0]
    assert "74 -> 128 -> 64 -> 32" in msg


def test_dense_head_says_so_when_residuals_are_off(caplog):
    with caplog.at_level("INFO", logger="swafi.impact_cnn_model"):
        _build(["--nb-dense-layers", "2", "--no-use-residual-dense"])
    msg = [r.getMessage() for r in caplog.records if "Dense head" in r.getMessage()][0]
    assert "no residual connections" in msg


def test_dense_head_singular_for_one_layer(caplog):
    with caplog.at_level("INFO", logger="swafi.impact_cnn_model"):
        _build(["--nb-dense-layers", "1", "--nb-dense-units", "128"])
    msg = [r.getMessage() for r in caplog.records if "Dense head" in r.getMessage()][0]
    assert "1 residual skip (" in msg


# --- soft-argmax pooling ----------------------------------------------------


def _spike_at(position, t_len=T_LEN, channels=3, height=1.0):
    """A single peak at one time step, zeros everywhere else."""
    x = np.zeros((1, t_len, channels), dtype="float32")
    x[0, position, :] = height
    return x


def test_softargmax_emits_peak_then_position():
    x = np.random.rand(4, T_LEN, 5).astype("float32")
    got = SoftArgmaxPooling(beta=1.0)(tf.constant(x)).numpy()
    assert got.shape == (4, 10)
    # First half is the plain maximum: Phase H showed softening it costs 23%.
    np.testing.assert_allclose(got[:, :5], x.max(axis=1), atol=1e-6)
    # Second half is a position, so it lives in [0, 1].
    assert (got[:, 5:] >= 0.0).all() and (got[:, 5:] <= 1.0).all()


@pytest.mark.parametrize("position,expected", [
    (0, 0.0),
    (T_LEN // 2, 0.5),
    (T_LEN - 1, 1.0),
])
def test_softargmax_locates_a_sharp_peak(position, expected):
    """With a sharp temperature the reported position is the peak's own."""
    got = SoftArgmaxPooling(beta=50.0)(tf.constant(_spike_at(position))).numpy()
    assert got[0, 3] == pytest.approx(expected, abs=0.02)


def test_softargmax_position_is_monotone_in_peak_time():
    layer = SoftArgmaxPooling(beta=50.0)
    positions = [layer(tf.constant(_spike_at(p))).numpy()[0, 3]
                 for p in (2, 12, 24, 36, 46)]
    assert all(b > a for a, b in zip(positions, positions[1:]))


def test_softargmax_position_is_resolution_independent():
    """Normalised, so the same peak fraction reports the same number."""
    a = SoftArgmaxPooling(beta=50.0)(
        tf.constant(_spike_at(24, t_len=49))).numpy()[0, 3]
    b = SoftArgmaxPooling(beta=50.0)(
        tf.constant(_spike_at(288, t_len=577))).numpy()[0, 3]
    assert a == pytest.approx(b, abs=0.02)


def test_softargmax_flat_temperature_reports_the_window_centre():
    """A flat input carries no timing, and must not fake one."""
    x = np.ones((1, T_LEN, 2), dtype="float32")
    got = SoftArgmaxPooling(beta=1.0)(tf.constant(x)).numpy()
    assert got[0, 2] == pytest.approx(0.5, abs=1e-4)
    assert got[0, 3] == pytest.approx(0.5, abs=1e-4)


def test_softargmax_temperature_is_learnable_and_positive():
    layer = SoftArgmaxPooling(beta=8.0)
    layer.build((None, T_LEN, 4))
    assert len(layer.trainable_weights) == 1
    # Held in log space, so no value of the weight can flip the softmax into
    # selecting the minimum.
    assert float(tf.exp(layer.log_beta).numpy()) == pytest.approx(8.0, rel=1e-5)


def test_softargmax_costs_one_parameter_over_mean_max():
    """Matched for practical purposes; the extra weight is the temperature."""
    mean_max = _build(["--tcn-pooling", "mean_max"])
    soft = _build(["--tcn-pooling", "softargmax"])
    assert soft.count_params() == mean_max.count_params() + 1


def test_softargmax_beta_reaches_the_layer():
    """The flag has to be assigned in parse_args, not merely declared."""
    sys.argv = ["test"] + BASE_ARGS + ["--tcn-pooling", "softargmax",
                                       "--tcn-softargmax-beta", "8"]
    options = ImpactCnnOptions()
    options.parse_args()
    assert options.tcn_softargmax_beta == 8.0


def test_softargmax_survives_a_save_load_roundtrip(tmp_path):
    model = _build(["--tcn-pooling", "softargmax",
                    "--tcn-softargmax-beta", "8"])
    x3 = np.random.rand(4, T_LEN, 1, 1, 1).astype("float32")
    x1 = np.random.rand(4, 10).astype("float32")
    before = model.predict([x3, x1], verbose=0)

    path = tmp_path / "softargmax.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    np.testing.assert_allclose(before, reloaded.predict([x3, x1], verbose=0),
                               atol=1e-6)


# --- three-term pooling: accumulation, peak and position --------------------


def test_mean_softargmax_is_one_term_wider_than_softargmax():
    """Adding the mean back costs a third of the pooled width, not nothing."""
    two_term = _build(["--tcn-pooling", "softargmax"])
    three_term = _build(["--tcn-pooling", "mean_softargmax"])
    assert three_term.count_params() > two_term.count_params()


def test_three_term_poolings_are_matched_to_each_other():
    """msegmax and msoft differ only in how they encode position.

    Both emit 3 x filters, so the sweep can compare bucketed timing with
    continuous timing without a capacity difference between them. The single
    extra parameter is the soft-argmax temperature.
    """
    mean_segmax = _build(["--tcn-pooling", "mean_segmax",
                          "--tcn-nb-segments", "2"])
    mean_soft = _build(["--tcn-pooling", "mean_softargmax"])
    assert mean_soft.count_params() == mean_segmax.count_params() + 1


def test_mean_softargmax_keeps_the_accumulation_term():
    """The mean branch must actually be present, not silently dropped."""
    model = _build(["--tcn-pooling", "mean_softargmax"])
    names = set()

    def walk(layer):
        for sub in getattr(layer, "layers", []):
            names.add(sub.name)
            walk(sub)

    walk(model)
    assert "temporal_mean" in names
    assert "temporal_softargmax" in names
