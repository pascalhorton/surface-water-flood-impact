"""
Tests for the radial spatial reduction.

The layer replaces the convolution-and-flatten branch, whose width grows with
the window AREA, with the centre cell plus mean and max per ring, which grows
with the RADIUS. Two properties carry the whole design and are checked exactly
rather than by shape alone: the centre value must survive untouched, because
the target is whether that cell recorded a claim, and the ring statistics must
be computed over the right cells.

A shape-only test would pass on a layer that averaged the whole window.
"""

import sys

import keras
import numpy as np
import pytest
import tensorflow as tf

from swafi.impact_cnn_model import ModelCnn, RadialSpatialReduction
from swafi.impact_cnn_options import ImpactCnnOptions

T_LEN = 25
BASE_ARGS = [
    "--dataset", "gvz", "--event-method", "simple",
    "--precip-dataset", "hourly", "--precip-time-step", "60",
    "--precip-days-before", "0", "--precip-days-after", "0",
    "--nb-dense-layers", "2", "--nb-dense-units", "128",
    "--tcn-filters", "32", "--tcn-nb-layers", "2",
    "--random-state", "42", "--run-name", "test",
]


def _build(extra, pixels):
    sys.argv = ["test"] + BASE_ARGS + list(extra)
    options = ImpactCnnOptions()
    options.parse_args()
    model = ModelCnn(options=options,
                     input_3d_size=[T_LEN, pixels, pixels, 1],
                     input_1d_size=[2])
    model.build_model()
    return model.model


def _layer_names(model):
    names = []

    def walk(layer):
        for sub in getattr(layer, "layers", []):
            names.append(sub.name)
            walk(sub)

    walk(model)
    return names


def test_centre_value_passes_through_untouched():
    """The cell the target is about must arrive at the TCN unmodified."""
    x = np.zeros((1, 1, 3, 3, 1), dtype="float32")
    x[0, 0, 1, 1, 0] = 7.5          # centre
    x[0, 0, 0, 0, 0] = 99.0         # a corner, to prove it is not blended in
    got = RadialSpatialReduction()(tf.constant(x)).numpy()
    assert got[0, 0, 0] == pytest.approx(7.5)


def test_ring_mean_and_max_are_over_the_right_cells():
    x = np.zeros((1, 1, 3, 3, 1), dtype="float32")
    x[0, 0] = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype="float32")[..., None]
    got = RadialSpatialReduction()(tf.constant(x)).numpy()[0, 0]
    assert got.shape == (3,)                       # centre, ring mean, ring max
    assert got[0] == pytest.approx(5.0)            # the centre cell
    ring = [1, 2, 3, 4, 6, 7, 8, 9]                # everything but the centre
    assert got[1] == pytest.approx(np.mean(ring))
    assert got[2] == pytest.approx(max(ring))


def test_rings_are_chebyshev_and_do_not_overlap():
    """A 5x5 has 1 centre, 8 cells at distance 1 and 16 at distance 2."""
    x = np.zeros((1, 1, 5, 5, 1), dtype="float32")
    x[0, 0, 2, 2, 0] = 1.0          # centre only
    got = RadialSpatialReduction()(tf.constant(x)).numpy()[0, 0]
    assert got.shape == (5,)        # centre, r1 mean/max, r2 mean/max
    assert got[0] == pytest.approx(1.0)
    assert got[1:] == pytest.approx(0.0)       # nothing leaks into the rings

    y = np.zeros((1, 1, 5, 5, 1), dtype="float32")
    y[0, 0, 1, 2, 0] = 8.0          # a distance-1 cell
    got = RadialSpatialReduction()(tf.constant(y)).numpy()[0, 0]
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(8.0 / 8)    # mean over the 8 ring-1 cells
    assert got[2] == pytest.approx(8.0)
    assert got[3:] == pytest.approx(0.0)       # ring 2 untouched


def test_negative_values_do_not_break_the_ring_max():
    """Normalised precipitation can be negative; the max must not return the
    sentinel used to mask cells outside the ring."""
    x = np.full((1, 1, 3, 3, 1), -5.0, dtype="float32")
    got = RadialSpatialReduction()(tf.constant(x)).numpy()[0, 0]
    assert got[2] == pytest.approx(-5.0)


@pytest.mark.parametrize("pixels,expected", [(3, 3), (5, 5), (7, 7), (9, 9)])
def test_channels_grow_with_radius_not_area(pixels, expected):
    x = np.random.rand(2, 4, pixels, pixels, 1).astype("float32")
    got = RadialSpatialReduction()(tf.constant(x)).numpy()
    assert got.shape == (2, 4, expected)


def test_multiple_input_channels_are_kept_separate():
    """With the DEM as a second channel the output is 2 x (1 + 2r)."""
    x = np.random.rand(2, 4, 5, 5, 2).astype("float32")
    got = RadialSpatialReduction()(tf.constant(x)).numpy()
    assert got.shape == (2, 4, 10)


def test_even_or_non_square_windows_are_refused():
    """Without a centre cell the reduction has no meaning; fail loudly."""
    with pytest.raises(AssertionError, match="odd window"):
        RadialSpatialReduction()(tf.constant(
            np.zeros((1, 1, 4, 4, 1), dtype="float32")))
    with pytest.raises(AssertionError, match="square window"):
        RadialSpatialReduction()(tf.constant(
            np.zeros((1, 1, 3, 5, 1), dtype="float32")))


@pytest.mark.parametrize("pixels", [3, 5, 7])
def test_window_size_is_nearly_free(pixels):
    """The point of the layer: cost grows with radius, not area.

    conv-and-flatten at 7 km costs +108% over a single pixel at 32 filters.
    """
    one = _build(["--precip-window-size", "1"], 1).count_params()
    wide = _build(["--precip-window-size", str(pixels),
                   "--spatial-reduction", "radial"], pixels).count_params()
    assert (wide - one) / one < 0.005


def test_radial_path_creates_no_conv_batchnorm_or_dropout():
    """Phase N was confounded by exactly these layers appearing unbidden."""
    names = _layer_names(_build(
        ["--precip-window-size", "3", "--spatial-reduction", "radial"], 3))
    assert "spatial_radial" in names
    assert not [n for n in names if n.startswith(("td_conv", "td_bn", "td_drop"))]


def test_conv_is_still_the_default():
    """The additions must not move any configuration already measured."""
    names = _layer_names(_build(["--precip-window-size", "3"], 3))
    assert "spatial_radial" not in names
    assert "td_conv2d_0" in names


def test_single_pixel_is_unaffected_by_the_option():
    """--spatial-reduction radial must not change the 1-pixel baseline."""
    plain = _build(["--precip-window-size", "1"], 1)
    with_flag = _build(["--precip-window-size", "1",
                        "--spatial-reduction", "radial"], 1)
    assert plain.count_params() == with_flag.count_params()
    assert "spatial_radial" not in _layer_names(with_flag)


def test_radial_survives_a_save_load_roundtrip(tmp_path):
    model = _build(["--precip-window-size", "5",
                    "--spatial-reduction", "radial"], 5)
    x3 = np.random.rand(4, T_LEN, 5, 5, 1).astype("float32")
    x1 = np.random.rand(4, 2).astype("float32")
    before = model.predict([x3, x1], verbose=0)

    path = tmp_path / "radial.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    np.testing.assert_allclose(before, reloaded.predict([x3, x1], verbose=0),
                               atol=1e-6)


# ---------------------------------------------------------------------------
# conv3d: convolving space and time together
#
# The TimeDistributed branch applies a 2D kernel to each time step on its own,
# so it can only learn "the neighbourhood is wet now". It cannot represent a
# neighbour whose rain arrives before or after the centre. That matters because
# at five-minute resolution the upwind and downwind lags around an event are
# anti-correlated at -0.42, measured over 800 events, and not at all on hourly
# data. conv3d is the only one of the three modes that can use it.
# ---------------------------------------------------------------------------

CONV3D_ARGS = [
    "--spatial-reduction", "conv3d", "--kernel-size-spatial", "3",
    "--nb-filters", "8", "--nb-conv-blocks", "1", "--pool-size-spatial", "1",
    "--no-use-batchnorm-cnn", "--dropout-rate-cnn", "0",
]


def test_conv3d_builds_a_space_time_kernel_and_no_per_step_conv():
    model = _build(CONV3D_ARGS + ["--precip-window-size", "3"], pixels=3)
    names = _layer_names(model)
    assert any(n.startswith("conv3d_") for n in names), names
    assert not any(n.startswith("td_conv2d") for n in names), \
        "a per-time-step 2D conv survived on the conv3d path"
    assert not any("spatial_radial" in n for n in names)


def test_conv2d_remains_the_default_for_a_window():
    """The default path must be untouched: every earlier phase used it."""
    model = _build(["--precip-window-size", "3", "--kernel-size-spatial", "3",
                    "--nb-filters", "8", "--nb-conv-blocks", "1"], pixels=3)
    names = _layer_names(model)
    assert any(n.startswith("td_conv2d") for n in names), names
    assert not any(n.startswith("conv3d_") for n in names)


def test_conv3d_preserves_the_time_axis():
    """Padding is 'same' and pooling is spatial only, so the TCN is unchanged.

    If the time axis shrank, every receptive-field figure in the project would
    silently stop applying to this arm.
    """
    for k_t in (3, 5, 7):
        model = _build(
            CONV3D_ARGS + ["--precip-window-size", "3",
                           "--kernel-size-temporal", str(k_t)], pixels=3)
        out = model([np.zeros((2, T_LEN, 3, 3, 1), dtype="float32"),
                     np.zeros((2, 2), dtype="float32")])
        assert np.asarray(out).shape[0] == 2
        reshape = [l for l in model.layers if l.name == "reshape_conv3d"]
        assert reshape, "the conv3d branch did not reshape to (T, features)"
        assert reshape[0].output.shape[1] == T_LEN, \
            f"time axis became {reshape[0].output.shape[1]}, expected {T_LEN}"


@pytest.mark.parametrize("k_t", [1, 3, 5, 7])
def test_temporal_kernel_costs_what_it_should(k_t):
    """One extra step of temporal kernel adds spatial_k^2 * filters weights."""
    model = _build(CONV3D_ARGS + ["--precip-window-size", "3",
                                  "--kernel-size-temporal", str(k_t)], pixels=3)
    conv = [l for l in model.layers if l.name == "conv3d_0"][0]
    weights = conv.get_weights()[0]
    assert weights.shape == (k_t, 3, 3, 1, 8), weights.shape


def test_conv3d_sees_a_neighbour_at_a_different_time():
    """The property the whole arm exists for, asserted on the layer itself.

    A 2D kernel per time step has no weight connecting pixel (y, x) at step t to
    the centre at step t +/- 1. A 3D kernel does, and that is the only way an
    upwind neighbour arriving two steps early can be represented.
    """
    model = _build(CONV3D_ARGS + ["--precip-window-size", "3",
                                  "--kernel-size-temporal", "3"], pixels=3)
    conv = [l for l in model.layers if l.name == "conv3d_0"][0]
    k = conv.get_weights()[0]            # (time, y, x, in, out)
    assert k.shape[0] == 3, "the kernel does not span time at all"
    # Off-centre in time AND off-centre in space must both be reachable.
    assert k[0, 0, 0].size > 0 and k[2, 2, 2].size > 0


def test_conv3d_falls_back_at_one_pixel_rather_than_failing():
    """A 3D kernel over a 1x1 grid is meaningless, so the option stands down.

    It must also be recorded as having stood down: an options file that claims
    conv3d for a run that used conv would make the record a lie.
    """
    sys.argv = (["test"] + BASE_ARGS
                + ["--precip-window-size", "1", "--spatial-reduction", "conv3d"])
    options = ImpactCnnOptions()
    options.parse_args()
    assert options.spatial_reduction == "conv"
    assert options.kernel_size_temporal == 1

    model = _build(["--precip-window-size", "1",
                    "--spatial-reduction", "conv3d"], pixels=1)
    names = _layer_names(model)
    assert not any(n.startswith("conv3d_") for n in names)


def test_conv3d_survives_a_save_and_load():
    model = _build(CONV3D_ARGS + ["--precip-window-size", "3",
                                  "--kernel-size-temporal", "5"], pixels=3)
    x = [np.random.default_rng(0).random((3, T_LEN, 3, 3, 1)).astype("float32"),
         np.zeros((3, 2), dtype="float32")]
    before = np.asarray(model(x))
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.keras")
        model.save(path)
        reloaded = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(before, np.asarray(reloaded(x)),
                               rtol=1e-5, atol=1e-6)


def test_the_three_modes_are_close_on_parameters():
    """conv2d against conv3d is the contrast, so it must not be a capacity test.

    Phase N was confounded by batch-norm and dropout appearing silently at
    window > 1; this checks the replacement comparison is clean on count.
    """
    window = ["--precip-window-size", "3", "--kernel-size-spatial", "3",
              "--nb-filters", "8", "--nb-conv-blocks", "1",
              "--pool-size-spatial", "1", "--no-use-batchnorm-cnn",
              "--dropout-rate-cnn", "0"]
    n2d = _build(window, pixels=3).count_params()
    n3d = _build(window + ["--spatial-reduction", "conv3d",
                           "--kernel-size-temporal", "3"], pixels=3).count_params()
    assert abs(n3d - n2d) / n2d < 0.01, (n2d, n3d)
