"""
Tests for the publication of the derived precipitation store.

The store is materialised once per (resolution, time step, period) and reused
across runs. Building it is the expensive part of moving to a finer time step -
resampling every step of the whole domain - so two things must hold:

  * a completed store is never rebuilt, including one built before the store was
    published by rename, which is recognised by its marker;
  * two runs that need the same store at the same time cannot damage it.

The second one is not hypothetical. A sweep with two arms at one resolution,
started in parallel, has both processes reach this code with no marker present.
Under the previous scheme both wrote to the same directory with mode='w' and the
first to finish published a marker over a store the other was still rewriting.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip_archive import PrecipitationArchive


def make_archive(tmp_path, time_step=0.5):
    """A minimal archive whose base data is 5-minute, over a small grid."""
    times = pd.date_range('2020-01-01 00:05', periods=288, freq='5min')
    vals = np.ones((288, 4, 4), dtype='float32')
    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset({'precip': (('time', 'y', 'x'), vals)},
                        coords={'time': times,
                                'y': np.arange(4.0), 'x': np.arange(4.0)})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p.dataset_name = 'TestArchive'
    p.tmp_dir = tmp_path
    p.resolution = 1
    p.native_time_step = 5 / 60
    p.time_step = 5 / 60
    return p


def store_paths(tmp_path):
    stores = sorted(q for q in tmp_path.iterdir()
                    if q.is_dir() and q.name.endswith('.zarr'))
    markers = sorted(tmp_path.glob('*.zarr.done'))
    building = sorted(q for q in tmp_path.iterdir() if '.building-' in q.name)
    return stores, markers, building


def test_build_publishes_one_store_and_no_leftovers(tmp_path):
    p = make_archive(tmp_path)
    p._use_derived_store(resolution=1, time_step=0.5)

    stores, markers, building = store_paths(tmp_path)
    assert len(stores) == 1
    assert len(markers) == 1
    assert building == [], "the private build directory was not cleaned up"
    # 30-minute bins summing six 5-minute steps of 1 mm.
    np.testing.assert_allclose(p.data['precip'].to_numpy(), 6.0)


def test_completed_store_is_not_rebuilt(tmp_path):
    p = make_archive(tmp_path)
    p._use_derived_store(resolution=1, time_step=0.5)
    stores, _, _ = store_paths(tmp_path)

    # A second run must reuse it rather than resample again, which is what makes
    # the later seeds of a sweep cheap.
    q = make_archive(tmp_path)
    q._resample = lambda data: pytest.fail("rebuilt a store that was complete")
    q._use_derived_store(resolution=1, time_step=0.5)

    assert store_paths(tmp_path)[0] == stores
    np.testing.assert_allclose(q.data['precip'].to_numpy(), 6.0)


def test_loser_of_a_race_keeps_the_winners_store(tmp_path):
    """Two runs build the same store; the first to publish wins, intact."""
    p = make_archive(tmp_path)
    p._use_derived_store(resolution=1, time_step=0.5)
    stores, markers, _ = store_paths(tmp_path)
    winner = stores[0]
    published_at = winner.stat().st_mtime_ns

    # A second run that finished a moment later, still holding its own copy.
    late = tmp_path / (winner.name + '.building-999-deadbeef')
    late.mkdir()
    (late / 'zarr.json').write_text('{"not": "the winner"}', encoding='utf-8')

    PrecipitationArchive._publish_derived_store(late, winner, markers[0])

    assert winner.stat().st_mtime_ns == published_at, "the winner was replaced"
    assert late.exists(), "the loser's copy is the caller's to clean up"
    assert not (winner / 'zarr.json').read_text(encoding='utf-8').startswith(
        '{"not"')


def test_interrupted_build_is_replaced_not_reused(tmp_path):
    """A directory with no marker is wreckage, and must not be published over."""
    p = make_archive(tmp_path)
    name = 'precip_testarchive_r1_t30min_2020-2020.zarr'
    partial = tmp_path / name
    partial.mkdir()
    (partial / 'stale.txt').write_text('half a store', encoding='utf-8')

    p._use_derived_store(resolution=1, time_step=0.5)

    stores, markers, building = store_paths(tmp_path)
    assert len(stores) == 1 and len(markers) == 1
    assert building == []
    assert not (tmp_path / name / 'stale.txt').exists()
    np.testing.assert_allclose(p.data['precip'].to_numpy(), 6.0)


def test_failed_build_leaves_nothing_behind(tmp_path):
    """A build that dies must not leave a partial store where a reader sees it.

    Without the rename this was the dangerous case: mode='w' wrote directly to
    the final path, so an interrupted build left a half-written store there.
    """
    p = make_archive(tmp_path)

    def explode(data):
        raise RuntimeError("out of memory, say")

    p._resample = explode
    with pytest.raises(RuntimeError):
        p._use_derived_store(resolution=1, time_step=0.5)

    stores, markers, building = store_paths(tmp_path)
    assert stores == [] and markers == []
    assert building == [], "a private build directory survived the failure"


def test_native_time_step_builds_no_store(tmp_path):
    """The 5-minute arms need no derived store, which is why they start first."""
    p = make_archive(tmp_path)
    p._use_derived_store(resolution=1, time_step=5 / 60)

    assert store_paths(tmp_path) == ([], [], [])


def test_the_build_never_writes_to_the_published_path(tmp_path):
    """The invariant the fix rests on, stated directly.

    Readers accept a store as soon as its marker exists, so the only way two
    runs can be safe is if neither ever writes into the path the other reads.
    """
    p = make_archive(tmp_path)
    seen = {}
    real = PrecipitationArchive._publish_derived_store

    def spy(building, derived_path, done_marker):
        seen['building'] = building
        seen['derived_path'] = derived_path
        # Nothing may be visible under the published name until this point.
        assert not derived_path.exists(), \
            "the store was written to the path readers use"
        return real(building, derived_path, done_marker)

    p._publish_derived_store = spy
    p._use_derived_store(resolution=1, time_step=0.5)

    assert seen['building'] != seen['derived_path']
    assert seen['building'].name.startswith(seen['derived_path'].name)
    assert seen['derived_path'].exists()
