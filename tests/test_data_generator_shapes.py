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


# --- the window start -------------------------------------------------------


def _window_length(days_before, days_after, hours_before, time_step=60):
    """Time steps in the extracted window, without touching any real data."""
    gen = ImpactCnnDataGenerator.__new__(ImpactCnnDataGenerator)
    gen.X_precip = object()
    gen.time_dim_size = None
    gen.precip_time_step = time_step
    gen.precip_days_before = days_before
    gen.precip_days_after = days_after
    gen.precip_hours_before = hours_before
    return gen.get_time_dim_size()


def test_hours_before_defaults_to_no_change():
    """Every configuration run before this option existed must be untouched."""
    assert _window_length(1, 0, 0) == 49
    assert _window_length(0, 0, 0) == 25
    assert _window_length(2, 1, 0) == 97


@pytest.mark.parametrize("days_before,hours_before,expected", [
    (0, 6, 31),
    (0, 12, 37),
    (1, 6, 55),
])
def test_hours_before_extends_the_window(days_before, hours_before, expected):
    assert _window_length(days_before, 0, hours_before) == expected


def test_hours_before_must_be_a_whole_number_of_steps():
    """A partial step would silently misalign the series against the events."""
    with pytest.raises(AssertionError, match="whole number"):
        _window_length(0, 0, 1, time_step=90)


def test_thirty_hour_window_splits_at_the_diurnal_minimum():
    """Why 6 hours and not some other number.

    The event definition is day-based, so with a whole-day window the two-way
    pooling split lands on midnight. Claims follow the convective cycle and peak
    at 18:00, with 7% of them in hour 0 of the event day - evening storms that
    ran over and were split into a second event. Six extra hours moves the split
    to 09:00, the quietest part of the day, which separates one convective day
    from the next instead of cutting through a storm.
    """
    t_len = _window_length(0, 0, 6)
    midnight = 6                      # index of e_date in the window
    split = t_len // 2                # SegmentMaxPooling boundary for n = 2
    assert split - midnight == 9      # 09:00 on the event day
