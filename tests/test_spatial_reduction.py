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
