import numpy as np
import pytest
from swafi.impact_dl_data_generator import ImpactDlDataGenerator
from swafi.impact_cnn_data_generator import ImpactCnnDataGenerator
from swafi.impact_tx_data_generator import ImpactTxDataGenerator


def make_event_props(n):
    # event_props: [date, x, y, cid]
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(n)], dtype='datetime64[D]')
    xs = np.zeros(n)
    ys = np.zeros(n)
    cids = np.zeros(n, dtype=int)
    ev = np.stack([dates.astype('datetime64[D]').astype(object), xs.astype(object), ys.astype(object), cids.astype(object)], axis=1)
    return ev


def test_impact_dl_generator_batch_building_is_abstract():
    """The base class holds the shared bookkeeping, not the batch assembly.

    _generate_batch lives in the subclasses because the inputs differ (a 3D
    precipitation block for the CNN, two series for the transformer), so the
    base class must refuse rather than half-work. The label-column shape this
    file cares about is checked on the concrete generators below.
    """
    n = 10
    event_props = make_event_props(n)
    x_static = np.zeros((n, 2))
    y = np.zeros(n)
    gen = ImpactDlDataGenerator(event_props, x_static, y, batch_size=4, shuffle=False,
                                mean_static=np.zeros(2), std_static=np.ones(2))

    with pytest.raises(NotImplementedError):
        gen._generate_batch(np.arange(4))

    # The public entry points delegate to it, so they must fail the same way.
    with pytest.raises(NotImplementedError):
        gen.get_batch_for_indices(np.arange(4))

    with pytest.raises(NotImplementedError):
        gen[0]


def test_impact_cnn_generator_returns_label_column():
    n = 8
    event_props = make_event_props(n)
    x_static = np.zeros((n, 3))
    # x_precip and x_dem can be None for this test
    y = np.ones(n)
    # Provide mean/std for static to avoid divide-by-zero during standardization
    gen = ImpactCnnDataGenerator(event_props, x_static, x_precip=None, x_dem=None, y=y, batch_size=2, shuffle=False,
                                 mean_static=np.zeros(3), std_static=np.ones(3))
    (x, xs), y_batch = gen.__getitem__(0)
    assert y_batch.ndim == 2 and y_batch.shape[1] == 1


def test_impact_tx_generator_returns_label_column():
    n = 6
    event_props = make_event_props(n)
    x_static = np.zeros((n, 1))
    y = np.array([0, 1, 0, 1, 0, 1])
    gen = ImpactTxDataGenerator(event_props, x_static, x_precip_hf=None, x_precip_daily=None, y=y, batch_size=3, shuffle=False,
                                mean_static=np.zeros(1), std_static=np.ones(1))
    (x_precip_daily, x_precip_hf, x_static_out), y_batch = gen.__getitem__(0)
    assert y_batch.ndim == 2 and y_batch.shape[1] == 1
