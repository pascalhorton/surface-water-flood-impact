"""
Tests for building the derived precipitation store in slabs.

A derived store is one resolution / time step of the base data, materialised
once and reused by every run that needs it. It used to be written with a single
lazy resample and one to_zarr call, which builds a dask graph of roughly a
thousand tasks per day of 5-minute input. Over eighteen years that graph was
large enough that the scheduler spent days optimising it before executing a
single task, so nothing was ever written.

Resampling a slab at a time bounds the graph. What these tests pin is that
bounding it changes nothing about the result: the slabbed store must hold the
same values, on the same time axis, as one resample of the whole series.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip_archive import PrecipitationArchive


def make_archive(tmp_path, days=4, ny=4, nx=4, seed=0):
    """A small archive whose base data is 5-minute, with reproducible values."""
    n = days * 288
    rng = np.random.default_rng(seed)
    times = pd.date_range('2020-01-01', periods=n, freq='5min')
    vals = rng.gamma(0.4, 2.0, size=(n, ny, nx)).astype('float32')
    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset({'precip': (('time', 'y', 'x'), vals)},
                        coords={'time': times,
                                'y': np.arange(float(ny)),
                                'x': np.arange(float(nx))})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p.dataset_name = 'TestArchive'
    p.tmp_dir = tmp_path
    p.resolution = 1
    p.native_time_step = 5 / 60
    p.time_step = 5 / 60
    return p


def reference(p, resolution, time_step):
    """What the single-shot build produced: one resample of everything."""
    q = PrecipitationArchive.__new__(PrecipitationArchive)
    q.__dict__.update(p.__dict__)
    q.resolution = resolution
    q.time_step = time_step
    return q._resample(q.data)


@pytest.mark.parametrize("minutes, slab_steps", [
    (30, 90 * 24 * 12),   # the production slab: one slab for this much data
    (30, 720),            # several slabs, boundaries inside the series
    (15, 720),            # a different ratio
    (60, 1440),           # the hourly target, two slabs
])
def test_slabbed_store_matches_a_single_resample(tmp_path, minutes, slab_steps):
    p = make_archive(tmp_path)
    p.DERIVED_STORE_SLAB_STEPS = slab_steps
    expected = reference(p, resolution=1, time_step=minutes / 60)

    p._use_derived_store(resolution=1, time_step=minutes / 60)

    got = p.data
    assert pd.DatetimeIndex(got['time'].values).equals(
        pd.DatetimeIndex(expected['time'].values))
    np.testing.assert_allclose(
        got['precip'].to_numpy(), expected['precip'].to_numpy(),
        rtol=0, atol=0,
        err_msg="slabbing changed the values, not just the task graph")


def test_more_slabs_give_the_same_store_as_fewer(tmp_path):
    """The slab length is a performance knob and must not touch the result."""
    a = make_archive(tmp_path / 'a')
    (tmp_path / 'a').mkdir()
    a.tmp_dir = tmp_path / 'a'
    a.DERIVED_STORE_SLAB_STEPS = 90 * 24 * 12
    a._use_derived_store(resolution=1, time_step=0.5)

    b = make_archive(tmp_path / 'b')
    (tmp_path / 'b').mkdir()
    b.tmp_dir = tmp_path / 'b'
    b.DERIVED_STORE_SLAB_STEPS = 720
    b._use_derived_store(resolution=1, time_step=0.5)

    np.testing.assert_array_equal(a.data['precip'].to_numpy(),
                                  b.data['precip'].to_numpy())


def test_slab_boundaries_fall_on_whole_chunks(tmp_path, monkeypatch):
    """No chunk may be written by two slabs, or the second read-modify-writes."""
    p = make_archive(tmp_path, days=8)
    p.DERIVED_STORE_SLAB_STEPS = 700  # deliberately not a multiple of anything
    offsets = []
    import swafi.precip_archive as mod
    real = mod.write_time_region

    def spy(zarr_path, values, t_offset):
        offsets.append((t_offset, values.shape[0]))
        return real(zarr_path, values, t_offset)

    monkeypatch.setattr(mod, 'write_time_region', spy)
    p._use_derived_store(resolution=1, time_step=0.5)

    assert offsets, "nothing was written"
    for start, length in offsets[:-1]:
        assert start % 720 == 0 and length % 720 == 0, \
            f"slab at {start} of length {length} straddles a chunk boundary"
    # Regions must tile the axis: contiguous, no gaps, no overlap.
    cursor = 0
    for start, length in offsets:
        assert start == cursor
        cursor += length
    assert cursor == p.data.sizes['time']


def test_spatial_coarsening_matches_too(tmp_path):
    p = make_archive(tmp_path, ny=8, nx=8)
    p.DERIVED_STORE_SLAB_STEPS = 720
    expected = reference(p, resolution=2, time_step=0.5)

    p._use_derived_store(resolution=2, time_step=0.5)

    np.testing.assert_allclose(p.data['precip'].to_numpy(),
                               expected['precip'].to_numpy(), rtol=0, atol=0)
    np.testing.assert_array_equal(p.data['y'].values, expected['y'].values)
    np.testing.assert_array_equal(p.data['x'].values, expected['x'].values)


def test_a_gapped_time_axis_is_refused_not_silently_shifted(tmp_path):
    """Slab cutting assumes a regular grid; say so rather than misplace data."""
    p = make_archive(tmp_path)
    keep = np.ones(p.data.sizes['time'], dtype=bool)
    keep[100:160] = False           # five hours missing from the middle
    p.data = p.data.isel(time=keep)

    with pytest.raises(AssertionError, match="gaps"):
        p._use_derived_store(resolution=1, time_step=0.5)
