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


def _window_length(hours_before, hours_after, time_step=60):
    """Time steps in the extracted window, without touching any real data."""
    gen = ImpactCnnDataGenerator.__new__(ImpactCnnDataGenerator)
    gen.X_precip = object()
    gen.time_dim_size = None
    gen.precip_time_step = time_step
    gen.precip_hours_before = hours_before
    gen.precip_hours_after = hours_after
    return gen.get_time_dim_size()


def test_the_windows_the_phases_ran_keep_their_length():
    """The configurations behind the published results, in hours.

    These were written in days, and the day-based arithmetic extracted one
    day more after the peak than it named, so the conversion is not a bare
    multiplication by 24 - which is the whole reason the options moved to
    hours. The lengths are unchanged.
    """
    # Phases K/Q/R, the standing configuration: days_before 0, days_after 0.
    assert _window_length(0, 24) == 25
    # Phase K arch_base: days_before 1, days_after 0.
    assert _window_length(24, 24) == 49
    # Phase K day30: days_before 0, hours_before 6, days_after 0.
    assert _window_length(6, 24) == 31
    # The old shipped default: days_before 2, days_after 1.
    assert _window_length(48, 48) == 97


@pytest.mark.parametrize("hours_before,hours_after,expected", [
    (6, 24, 31),
    (12, 24, 37),
    (30, 24, 55),
    (0, 12, 13),
    (12, 12, 25),
])
def test_each_side_of_the_window_counts(hours_before, hours_after, expected):
    assert _window_length(hours_before, hours_after) == expected


# --- the bounds themselves --------------------------------------------------


ANCHOR = np.datetime64('2021-06-15T00:00')  # e_date, the simple definition


def _bounds(hours_before, hours_after, time_step=60):
    gen = ImpactCnnDataGenerator.__new__(ImpactCnnDataGenerator)
    gen.precip_time_step = time_step
    gen.precip_hours_before = hours_before
    gen.precip_hours_after = hours_after
    return gen.get_precip_window_bounds(ANCHOR)


def test_the_hours_count_from_the_event_time():
    """The names and the extraction agree, which they did not before."""
    for hours in (0, 1, 6, 24, 72):
        t_start, t_end = _bounds(hours, hours)
        assert ANCHOR - t_start == np.timedelta64(hours, 'h')
        assert t_end - ANCHOR == np.timedelta64(hours, 'h')


def test_the_default_window_is_the_event_day():
    """0 h before and 24 h after midnight is the event day itself.

    Phase K's best window, and the standing configuration of Q and R. The
    intensity peak lies inside it by construction, since the simple event
    definition dates an event by the day its peak falls in.
    """
    t_start, t_end = _bounds(0, 24)
    assert t_start == ANCHOR
    assert t_end == np.datetime64('2021-06-16T00:00')


def test_the_window_need_not_be_symmetric():
    """The two sides move independently, which the evidence asks for.

    Phase K: the event day alone beat the same day plus the previous evening
    and plus the whole previous day, so the useful shapes are asymmetric.
    """
    t_start, t_end = _bounds(6, 24)
    assert t_start == np.datetime64('2021-06-14T18:00')   # previous evening
    assert t_end == np.datetime64('2021-06-16T00:00')


def test_each_side_moves_on_its_own():
    """Antecedent rain without an equally long tail after the damage."""
    plain_start, plain_end = _bounds(0, 24)
    long_start, long_end = _bounds(6, 24)
    assert plain_start - long_start == np.timedelta64(6, 'h')
    assert long_end == plain_end


def test_a_restored_generator_with_a_stale_length_is_refused():
    """A pickle written against the old meaning must not pass silently.

    Its stored length and its stored parameters disagree by a day, and the
    extraction would then hand back a short window that the caller pads or
    blanks without failing - so the input would quietly go empty.
    """
    gen = ImpactCnnDataGenerator.__new__(ImpactCnnDataGenerator)
    gen.X_precip = object()
    gen.precip_time_step = 60
    gen.precip_hours_before = 0
    gen.precip_hours_after = 24    # 25 steps
    gen.time_dim_size = 97         # what an older configuration extracted

    with pytest.raises(ValueError, match="hours"):
        gen.get_time_dim_size()


def test_hours_before_must_be_a_whole_number_of_steps():
    """A partial step would silently misalign the series against the events."""
    with pytest.raises(AssertionError, match="whole number"):
        _window_length(1, 24, time_step=90)


def test_thirty_hour_window_splits_at_the_diurnal_minimum():
    """Why 6 hours and not some other number.

    The window is anchored on midnight of the event day, so with a whole-day
    window the two-way pooling split lands on that midnight. Claims follow
    the convective cycle and peak at 18:00, with 7.2% of them in hour 0 of
    the event day - evening storms that ran over and were split into a second
    event. Six extra hours moves the split to 09:00, the quietest part of the
    day, which separates one convective day from the next instead of cutting
    through a storm.
    """
    t_len = _window_length(6, 24)
    midnight = 6                      # index of e_date in the window
    split = t_len // 2                # SegmentMaxPooling boundary for n = 2
    assert split - midnight == 9      # 09:00 on the event day


def _full_dataset_indices(gen):
    """Every index a full pass over the generator actually returns."""
    seen = []
    for i in range(gen.get_number_of_batches_for_full_dataset()):
        _, y_batch = gen.get_ordered_batch_from_full_dataset(i)
        seen.append(np.asarray(y_batch).reshape(-1))
    return np.concatenate(seen) if seen else np.array([])


def _labelled_generator(n, batch_size):
    """A generator whose label is its own index, so a pass is self-identifying."""
    event_props = make_event_props(n)
    x_static = np.zeros((n, 2))
    y = np.arange(n, dtype=float)
    return ImpactCnnDataGenerator(
        event_props, x_static, x_precip=None, x_dem=None, y=y,
        batch_size=batch_size, shuffle=False,
        mean_static=np.zeros(2), std_static=np.ones(2))


@pytest.mark.parametrize("n, batch_size", [
    (10, 4),    # a ragged last batch
    (12, 4),    # an exact multiple, where the last batch ends on the boundary
    (1, 4),     # a single sample, entirely inside one short batch
    (5, 1),     # one sample per batch
    (7, 100),   # one batch larger than the dataset
])
def test_full_dataset_pass_returns_every_sample_once(n, batch_size):
    """A pass over the full dataset must return all of it, in order.

    The clamp on the last batch used to be len(y) - 1, which silently dropped
    the final sample of every set. It cost one row in 282,048 on the validation
    split, so no score moved by anything visible, but assessment ran on a
    dataset it did not have.
    """
    gen = _labelled_generator(n, batch_size)
    seen = _full_dataset_indices(gen)

    assert len(seen) == n, f"a full pass returned {len(seen)} of {n} samples"
    np.testing.assert_array_equal(seen, np.arange(n))


def test_full_dataset_pass_includes_the_last_sample():
    """The regression itself, stated as its own case.

    The bug was invisible in aggregate metrics and only showed up as a length,
    so pin the specific property rather than trusting a mean to reveal it.
    """
    n = 10
    gen = _labelled_generator(n, batch_size=4)
    seen = _full_dataset_indices(gen)
    assert n - 1 in seen, "the final sample was never returned by any batch"


def test_full_dataset_batches_do_not_overlap():
    """Consecutive batches must tile the dataset, not overlap or leave gaps."""
    gen = _labelled_generator(23, batch_size=5)
    seen = _full_dataset_indices(gen)
    assert len(np.unique(seen)) == len(seen), "a sample was returned twice"
