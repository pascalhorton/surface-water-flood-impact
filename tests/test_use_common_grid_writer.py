import types

import numpy as np
import pandas as pd
import xarray as xr

from swafi.utils.use_common import GridPredictionWriter, predict_events_in_chunks


def _make_setup():
    ids_map = np.array([[0, 11, 12],
                        [13, 0, 14]], dtype=np.int32)
    domain = types.SimpleNamespace(cids={'ids_map': ids_map})
    time = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    pred = np.zeros((len(time), 2, 3), dtype=np.float32)
    ds_pred = xr.Dataset({'predict': (('time', 'y', 'x'), pred)},
                         coords={'time': time, 'x': [0., 1., 2.], 'y': [0., 1.]})
    return domain, ds_pred


def test_grid_writer_masks_and_writes_like_the_per_cell_loop():
    domain, ds_pred = _make_setup()
    writer = GridPredictionWriter(ds_pred, domain)

    assert writer.get_map_cids() == {11, 12, 13, 14}

    writer.mask_outside_domain()
    assert np.isnan(ds_pred['predict'].values[:, 0, 0]).all()
    assert np.isnan(ds_pred['predict'].values[:, 1, 1]).all()

    writer.fill_cells({12}, np.nan)
    assert np.isnan(ds_pred['predict'].values[:, 0, 2]).all()
    writer.fill_cells(set(), np.nan)  # no-op

    cids = [11, 11, 11, 13]
    dates = pd.to_datetime(['2020-01-03 14:00', '2020-01-03 16:00',
                            '2020-01-05 02:00', '2020-01-02 23:00'])
    values = [0.4, 0.7, 0.0, 1.0]
    writer.write_events(cids, dates, values)

    # Same-day collision: the last nonzero write wins; zero predictions are
    # skipped (like the original per-event loops).
    assert ds_pred['predict'].values[2, 0, 1] == np.float32(0.7)
    assert ds_pred['predict'].values[4, 0, 1] == 0.0
    assert ds_pred['predict'].values[1, 1, 0] == 1.0
    # Untouched cell keeps its background
    assert (ds_pred['predict'].values[:, 1, 2] == 0.0).all()


def test_predict_events_in_chunks_batches_all_indices():
    calls = []

    class _FakeGenerator:
        event_props = np.arange(40).reshape(10, 4)

        def get_batch_for_indices(self, idxs):
            calls.append(len(idxs))
            return np.asarray(idxs, dtype=float).reshape(-1, 1), None

    class _FakeModel:
        def predict(self, x, verbose=0):
            return x * 2.0

    idxs = np.arange(10)
    y_pred = predict_events_in_chunks(_FakeModel(), _FakeGenerator(), idxs,
                                      chunk_size=4)
    np.testing.assert_allclose(y_pred, idxs * 2.0)
    assert calls == [4, 4, 2]
