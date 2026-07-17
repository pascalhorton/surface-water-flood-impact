"""
A/B tests for the optimized claims-events linking pipeline: every fast path is
compared against a brute-force reference implementing the previous semantics.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from swafi.damages import Damages
from swafi.events import Events
from swafi.utils.event_extraction import _stamp_precip_dataset


def _bare_damages():
    d = object.__new__(Damages)
    d.name = 'test'
    d.use_dump = False
    d.pickles_dir = '.'
    d.mask = dict(extent=(None, None, None, None), shape=None,
                  mask=np.array([]), xs=np.array([]), ys=np.array([]))
    d.claim_categories = []
    d.exposure_categories = []
    d.selected_claim_categories = []
    d.selected_exposure_categories = []
    return d


def _random_simple_events(n_events=20000, n_cids=60, seed=0):
    rng = np.random.default_rng(seed)
    e_date = (pd.Timestamp('2020-01-01')
              + pd.to_timedelta(rng.integers(0, 3 * 365, n_events), unit='D'))
    return pd.DataFrame({
        'eid': np.arange(1, n_events + 1),
        'cid': rng.integers(1, n_cids + 1, n_events),
        'e_date': e_date,
        'i_max_date': e_date + pd.to_timedelta(
            rng.integers(-8 * 60, 26 * 60, n_events), unit='min'),
        'i_max': rng.random(n_events) * 100,
    })


def _random_claims(n_claims=250, n_cids=60, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'cid': rng.integers(1, n_cids + 1, n_claims),
        'date_claim': pd.Timestamp('2020-01-01') + pd.to_timedelta(
            rng.integers(0, 3 * 365, n_claims), unit='D'),
        'selection': rng.integers(1, 5, n_claims),
    })


def _brute_simple_candidates(events_df, claim):
    """Previous full-scan candidate lookup for the simple method."""
    mask = ((events_df['cid'] == claim['cid'])
            & (events_df['i_max_date'] >= claim['date_claim'] - pd.Timedelta(hours=8))
            & (events_df['i_max_date'] <= claim['date_claim'] + pd.Timedelta(hours=26)))
    sub = events_df[mask]
    return None if sub.empty else sub


def test_simple_candidates_match_brute_force():
    events_df = _random_simple_events()
    claims = _random_claims()
    events_by_cid = Damages._group_events_by_cid(events_df, 'i_max_date')

    n_with_candidates = 0
    for _, claim in claims.iterrows():
        ref = _brute_simple_candidates(events_df, claim)
        new = Damages._get_potential_simple_events(claim, events_by_cid)
        if ref is None:
            assert new is None
        else:
            n_with_candidates += 1
            assert sorted(new['eid']) == sorted(ref['eid'])
    assert n_with_candidates > 50  # the test must exercise non-trivial cases


def _brute_classic_candidates(events_df, claim, window_days):
    """Previous full-scan candidate lookup for the classic method."""
    window_days = sorted(window_days, reverse=True)
    date_window_end, date_window_start = Damages._get_window_dates(
        claim['date_claim'], window_days[0])
    sub = events_df[
        (events_df['cid'] == claim['cid'])
        & (events_df['e_start'] < date_window_end)
        & (events_df['e_end'] > date_window_start)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub['min_window'] = window_days[0]
    for window in window_days[1:]:
        date_window_end, date_window_start = Damages._get_window_dates(
            claim['date_claim'], window)
        sub.loc[(sub['e_start'] < date_window_end)
                & (sub['e_end'] > date_window_start), 'min_window'] = window
    return sub


def test_classic_candidates_match_brute_force():
    rng = np.random.default_rng(2)
    n = 20000
    e_start = (pd.Timestamp('2020-01-01')
               + pd.to_timedelta(rng.integers(0, 3 * 365 * 24, n), unit='h'))
    events_df = pd.DataFrame({
        'eid': np.arange(1, n + 1),
        'cid': rng.integers(1, 61, n),
        'e_start': e_start,
        'e_end': e_start + pd.to_timedelta(rng.integers(1, 96, n), unit='h'),
    })
    claims = _random_claims(seed=3)
    window_days = [5, 3, 1]

    events_by_cid = Damages._group_events_by_cid(events_df, 'e_start')
    max_duration = (events_df['e_end'] - events_df['e_start']).max()

    n_with_candidates = 0
    for _, claim in claims.iterrows():
        ref = _brute_classic_candidates(events_df, claim, window_days)
        new = Damages._get_potential_classic_events(
            claim, events_by_cid, sorted(window_days, reverse=True), max_duration)
        if ref is None:
            assert new is None
        else:
            n_with_candidates += 1
            ref = ref.sort_values('eid')
            new = new.sort_values('eid')
            assert new['eid'].tolist() == ref['eid'].tolist()
            assert new['min_window'].tolist() == ref['min_window'].tolist()
    assert n_with_candidates > 50


def _ref_best_simple_eid(candidates, claim):
    """Previous best-candidate selection for the simple method."""
    if len(candidates) == 1:
        return candidates.iloc[0].eid
    best = candidates[candidates['i_max'] == candidates['i_max'].max()]
    if len(best) == 1:
        return best.iloc[0].eid
    same_day = best[best['e_date'] == claim['date_claim'].floor('D')]
    if same_day.empty:
        date_diff = (best['e_date'] - claim['date_claim']).abs()
        return best.loc[date_diff.idxmin()].eid
    return same_day.iloc[0].eid


def test_simple_link_end_to_end_matches_reference():
    events_df = _random_simple_events(n_events=8000, n_cids=30, seed=4)
    claims = _random_claims(n_claims=200, n_cids=40, seed=5)  # some cids unmatched

    # Reference results
    expected_eid = {}
    expected_remove = []
    for i, claim in claims.iterrows():
        cands = _brute_simple_candidates(events_df, claim)
        if cands is None:
            continue
        best = _ref_best_simple_eid(cands, claim)
        expected_eid[i] = best
        if len(cands) > 1:
            expected_remove.extend(e for e in cands['eid'] if e != best)
    expected_remove = [ev for ev in expected_remove
                       if ev not in set(expected_eid.values())]

    damages = _bare_damages()
    damages.claims = claims.copy()
    events_obj = SimpleNamespace(events=events_df)
    events_to_remove = damages.link_with_events(events_obj, method='simple',
                                                filename='unused.pickle')

    # Unmatched claims must be dropped (regression: eid stayed NaN before)
    assert len(damages.claims) == len(expected_eid)
    assert pd.api.types.is_integer_dtype(damages.claims['eid'])
    assert (damages.claims['eid'] != 0).all()

    # Matched eids equal the reference (compare via original claim identity)
    result = dict(zip(
        zip(claims.loc[list(expected_eid)].cid, claims.loc[list(expected_eid)].date_claim),
        expected_eid.values()))
    for _, row in damages.claims.iterrows():
        assert result[(row['cid'], row['date_claim'])] == row['eid']

    assert sorted(events_to_remove) == sorted(expected_remove)


def test_simple_link_tie_breaking():
    # Three events with identical i_max; the one on the claim day must win
    claim_day = pd.Timestamp('2021-06-10')
    events_df = pd.DataFrame({
        'eid': [1, 2, 3],
        'cid': [7, 7, 7],
        'e_date': [claim_day - pd.Timedelta(days=1), claim_day,
                   claim_day + pd.Timedelta(days=1)],
        'i_max_date': [claim_day - pd.Timedelta(hours=6), claim_day,
                       claim_day + pd.Timedelta(hours=20)],
        'i_max': [50.0, 50.0, 50.0],
    })
    damages = _bare_damages()
    damages.claims = pd.DataFrame({
        'cid': [7], 'date_claim': [claim_day], 'selection': [1]})
    damages.link_with_events(SimpleNamespace(events=events_df),
                             method='simple', filename='unused.pickle')
    assert damages.claims.iloc[0]['eid'] == 2


def test_get_events_for_removed_claims_matches_brute_force():
    events_df = _random_simple_events(n_events=8000, n_cids=30, seed=6)
    removed_claims = _random_claims(n_claims=150, n_cids=30, seed=7)
    linked_eids = list(np.random.default_rng(8).integers(1, 8000, 500))

    events = Events(use_dump=False)
    events.events = events_df
    damages = SimpleNamespace(claims=pd.DataFrame({'eid': linked_eids}))

    # Brute-force reference (previous implementation)
    sub = events_df[events_df['cid'].isin(removed_claims['cid'].unique())].copy()
    sub['mid_date'] = sub['e_date']
    expected = []
    for _, claim in removed_claims.iterrows():
        mask = ((sub['cid'] == claim['cid'])
                & (sub['mid_date'] >= claim['date_claim'] - pd.Timedelta(days=1))
                & (sub['mid_date'] <= claim['date_claim'] + pd.Timedelta(days=1)))
        expected.extend(sub.loc[mask, 'eid'].tolist())
    expected = [ev for ev in expected if ev not in set(linked_eids)]

    result = events.get_events_for_removed_claims(removed_claims, damages)
    assert sorted(result) == sorted(expected)


def test_select_locations_with_contracts_e_date_and_empty_cells():
    # Simple-method events (only e_date) must work, and (cid, year) pairs
    # without contracts must be removed
    events = Events(use_dump=False)
    events.events = pd.DataFrame({
        'eid': [1, 2, 3, 4, 5],
        'cid': [10, 10, 20, 30, 40],
        'e_date': pd.to_datetime(['2020-05-01', '2021-05-01', '2020-07-01',
                                  '2021-08-01', '2020-09-01']),
    })
    cids_list = np.array([10.0, 20.0, 30.0])
    exposure = pd.DataFrame({
        'mask_index': [0, 0, 1, 2],
        'year': [2020, 2021, 2020, 2021],
        'selection': [5, 0, 3, 0],  # cid 10 in 2021 and cid 30 in 2021 empty
    })
    damages = SimpleNamespace(cids_list=cids_list, exposure=exposure)

    events.select_locations_with_contracts(damages)

    # cid 40 not in cids_list; (10, 2021) and (30, 2021) removed
    assert sorted(events.events['eid']) == [1, 3]


def test_create_cids_list_matches_brute_force():
    rng = np.random.default_rng(9)
    nx, ny = 12, 9
    xs_axis = 2600500.0 + np.arange(nx) * 1000.0
    ys_axis = 1200500.0 - np.arange(ny) * 1000.0
    xs_grid, ys_grid = np.meshgrid(xs_axis, ys_axis)
    ids_map = np.arange(1, nx * ny + 1).reshape(ny, nx)

    mask = rng.random((ny, nx)) > 0.6
    mask[0, 0] = True

    d = _bare_damages()
    d.mask['mask'] = mask
    d.mask['xs'] = xs_grid
    d.mask['ys'] = ys_grid
    d.domain = SimpleNamespace(cids={'xs': xs_grid, 'ys': ys_grid,
                                     'ids_map': ids_map})

    d._create_cids_list()

    # Brute-force reference (previous per-cell loop)
    xs_mask = np.extract(mask, xs_grid)
    ys_mask = np.extract(mask, ys_grid)
    expected = np.array([
        ids_map[ys_axis == y, xs_axis == x][0]
        for x, y in zip(xs_mask, ys_mask)], dtype=float)

    np.testing.assert_array_equal(d.cids_list, expected)


def test_to_xarray_matches_reference():
    xs = 2600500.0 + np.arange(10) * 1000.0
    ys = 1200500.0 - np.arange(8) * 1000.0
    time = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    rng = np.random.default_rng(10)

    d = _bare_damages()
    d.year_start = 2021
    d.year_end = 2021
    d.domain = SimpleNamespace(get_x_axis=lambda: xs, get_y_axis=lambda: ys)
    d.exposure = pd.DataFrame({
        'year': [2021] * 6,
        'x': rng.choice(xs, 6),
        'y': rng.choice(ys, 6),
        'selection': [4, 2, 0, 7, 1, 3],  # one row without exposure
    })
    d.claims = pd.DataFrame({
        'date_claim': pd.to_datetime(['2021-06-05', '2021-07-10', '2021-08-15']),
        'x': rng.choice(xs, 3),
        'y': rng.choice(ys, 3),
        'selection': [1, 2, 3],
    })
    removed = pd.DataFrame({
        'date_claim': pd.to_datetime(['2021-03-03']),
        'x': [xs[4]], 'y': [ys[2]],
    })

    ds = d.to_xarray(save_to_nc=False, removed_claims=removed)

    # Brute-force reference (previous per-row loops)
    exp_ref = np.full((len(time), len(ys), len(xs)), np.nan, dtype=np.float32)
    cl_ref = np.full_like(exp_ref, np.nan)
    rm_ref = np.full_like(exp_ref, np.nan)
    for _, row in d.exposure.iterrows():
        if row['selection'] == 0:
            continue
        year_mask = (time.year == row['year'])
        x_idx = np.argmin(np.abs(xs - row['x']))
        y_idx = np.argmin(np.abs(ys - row['y']))
        exp_ref[year_mask, y_idx, x_idx] = row['selection']
        cl_ref[year_mask, y_idx, x_idx] = 0
    for _, row in d.claims.iterrows():
        t_idx = np.searchsorted(time, pd.to_datetime(row['date_claim']))
        x_idx = np.argmin(np.abs(xs - row['x']))
        y_idx = np.argmin(np.abs(ys - row['y']))
        cl_ref[t_idx, y_idx, x_idx] = row['selection']
    for _, row in removed.iterrows():
        t_idx = np.searchsorted(time, pd.to_datetime(row['date_claim']))
        x_idx = np.argmin(np.abs(xs - row['x']))
        y_idx = np.argmin(np.abs(ys - row['y']))
        rm_ref[t_idx, y_idx, x_idx] = 1

    np.testing.assert_array_equal(ds['exposure'].values, exp_ref)
    np.testing.assert_array_equal(ds['claims'].values, cl_ref)
    np.testing.assert_array_equal(ds['removed_claims'].values, rm_ref)


def test_extract_claims_from_grids_matches_per_slice():
    rng = np.random.default_rng(11)
    ny, nx, n_days = 7, 9, 40
    mask = rng.random((ny, nx)) > 0.5
    data = rng.integers(0, 3, (n_days, ny, nx)) * (rng.random((n_days, ny, nx)) > 0.7)
    dates = [datetime(2021, 1, 1 + i).date() for i in range(min(n_days, 30))]
    data = data[:len(dates)]

    d = _bare_damages()
    d.mask['mask'] = mask

    new = d._extract_claims_from_grids(data, dates, 'A')

    # Brute-force reference (previous per-date loop)
    rows = []
    for i_date, date in enumerate(dates):
        indices, values = d._extract_non_null_claims(data[i_date])
        for idx, val in zip(indices, values):
            rows.append((date, idx, val))
    ref = pd.DataFrame(rows, columns=['date_claim', 'mask_index', 'A'])

    assert len(new) == len(ref)
    assert new['date_claim'].tolist() == ref['date_claim'].tolist()
    assert new['mask_index'].tolist() == ref['mask_index'].tolist()
    assert new['A'].tolist() == ref['A'].tolist()


def test_stamp_precip_dataset_roundtrips_through_parquet(tmp_path):
    events = pd.DataFrame({'eid': [1, 2, 3], 'cid': [10, 20, 30]})
    events = _stamp_precip_dataset(events, '5min')

    assert isinstance(events['precip_dataset'].dtype, pd.CategoricalDtype)

    path = tmp_path / 'events.parquet'
    events.to_parquet(path)
    reloaded = pd.read_parquet(path)
    assert reloaded['precip_dataset'].unique().tolist() == ['5min']


def test_check_precip_dataset():
    events = Events(use_dump=False)

    # Legacy file without the provenance column: warning only, no raise
    events.events = pd.DataFrame({'eid': [1, 2]})
    events.check_precip_dataset('hourly')

    # Matching provenance: OK
    events.events = _stamp_precip_dataset(
        pd.DataFrame({'eid': [1, 2]}), 'hourly')
    events.check_precip_dataset('hourly')

    # Mismatch: raises with both names in the message
    events.events = _stamp_precip_dataset(
        pd.DataFrame({'eid': [1, 2]}), '5min')
    with pytest.raises(ValueError, match="5min.*hourly"):
        events.check_precip_dataset('hourly')
