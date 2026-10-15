"""
A trained model is a function of an event definition as much as of the weather.
The catalogue decides which cell-days are candidates at all, and the '*_q'
features say where each event sits in a distribution; both are fixed at
extraction time.

None of it is recorded inside the events file. The detection settings live in
the FILE NAME, and nowhere else - not in the parquet, not in the pickle the
linkage writes from it. So inference had two ways to agree with training: keep
its own copy of the settings, or read them off the name. It did neither. Four
of the five use_*.py scripts called get_events() with only the dataset name,
taking its defaults - a 1-hour detection window on the per-cell q98 - and
applied models trained on 12-hour, 10 mm, centred, peak-dated events to a
catalogue built another way. The fifth kept a copy, which could not express the
detection step at all. Nothing failed; the scores simply answered a different
question.

parse_detection_tag is the inverse of detection_tag, so the settings are read
back from the only place that holds them. These tests pin that the inversion is
exact over everything detection_tag can write, that a name it cannot read is
refused rather than guessed at, and that the test-event cache is keyed on the
full definition - including the detection step, whose absence from the key
would have let the pinned and unpinned 5-minute catalogues share one cache.
"""
import itertools

import pytest

from swafi.utils.event_extraction import detection_tag, parse_detection_tag

HOURLY = 1.0
FIVE_MIN = 5 / 60

# The two catalogues this project trains on.
HOURLY_NAME = ('events_cpc_hourly_model_domain_2005_2024_simple'
               '_det12h_thr10mm_centered_peakdays.parquet')
FIVE_MIN_NAME = ('events_cpc_5min_model_domain_2005_2024_simple'
                 '_det12h_thr10mm_centered_peakdays_step1h.parquet')


def test_the_training_catalogue_names_read_back_exactly():
    """The settings extract_precipitation_events.py was run with."""
    assert parse_detection_tag(HOURLY_NAME, HOURLY) == dict(
        detection_window_h=12.0, detection_threshold=10.0,
        detection_centered=True, detection_peak_days=True,
        detection_step_h=None)

    # The pinned 5-minute catalogue differs from the hourly one in exactly one
    # setting, and it is the one get_events could not express.
    assert parse_detection_tag(FIVE_MIN_NAME, FIVE_MIN) == dict(
        detection_window_h=12.0, detection_threshold=10.0,
        detection_centered=True, detection_peak_days=True,
        detection_step_h=1.0)


def test_the_unpinned_five_minute_catalogue_is_told_apart_from_the_pinned_one():
    """They differ by one tag, and by ~7 points of agreement with the hourly
    catalogue (92.8% against 85.7% on the real archive)."""
    unpinned = FIVE_MIN_NAME.replace('_step1h', '')

    assert parse_detection_tag(unpinned, FIVE_MIN)['detection_step_h'] is None
    assert parse_detection_tag(FIVE_MIN_NAME, FIVE_MIN)['detection_step_h'] == 1.0


@pytest.mark.parametrize("window_h, threshold, centered, peak_days, step_h", list(
    itertools.product([None, 0.5, 1.0, 3.0, 12.0], [None, 5.0, 10.0],
                      [True, False], [True, False], [None, 0.25, 1.0])))
@pytest.mark.parametrize("time_step_h", [HOURLY, FIVE_MIN])
def test_every_tag_the_writer_can_write_reads_back(window_h, threshold,
                                                   centered, peak_days, step_h,
                                                   time_step_h):
    """The inversion is exact, not best-effort.

    Read-back is not always equality of the arguments: centring and peak dating
    are deliberately left out of the tag for a single-step window, where they
    are no-ops, so the catalogue predating them is reused rather than re-run.
    What must hold is that the settings read back BUILD THE SAME CATALOGUE,
    which is what re-encoding to the same tag says.
    """
    tag = detection_tag(window_h, threshold, centered, peak_days, time_step_h,
                        step_h)
    settings = parse_detection_tag(f'events_x_simple{tag}.parquet', time_step_h)

    assert settings is not None
    assert detection_tag(time_step_h=time_step_h, **settings) == tag


@pytest.mark.parametrize("name", [
    'events_cpc_model_domain_2005_2024_classic.parquet',
    'events_cpc_hourly_model_domain_2005_2024_simple.parquet',
])
def test_a_name_without_a_detection_tag_reads_as_none(name):
    """The classic method, and the extractions predating the tag: the caller
    then uses the defaults, which are what built them."""
    assert parse_detection_tag(name, HOURLY) is None


def test_the_reference_saved_beside_a_catalogue_carries_its_tag():
    """The '_ref.pkl' companion must resolve to the same definition, or the
    '*_q' features would be ranked against another catalogue's distribution."""
    ref = FIVE_MIN_NAME.replace('.parquet', '_ref.pkl')

    assert parse_detection_tag(ref, FIVE_MIN) == parse_detection_tag(
        FIVE_MIN_NAME, FIVE_MIN)


def test_the_wrong_time_step_is_refused_rather_than_guessed_at():
    """A tag's meaning depends on the step of the data it belongs to: an
    hourly detection step is written for a 5-minute catalogue and omitted for
    an hourly one, so reading the 5-minute name as hourly would silently drop
    '_step1h' and hand back the unpinned definition."""
    with pytest.raises(ValueError, match="time_step_h"):
        parse_detection_tag(FIVE_MIN_NAME, HOURLY)


def test_a_tag_this_parser_does_not_understand_is_refused():
    with pytest.raises(ValueError):
        parse_detection_tag('events_x_simple_det12h_thr10mm_peakdays'
                            '_centered.parquet', HOURLY)


def test_the_test_event_cache_is_keyed_on_the_whole_definition():
    """Two definitions must never share a cache file.

    get_events names its pickle with detection_tag(), so the key is the tag.
    The detection step was missing from it, which is how a pinned 5-minute
    extraction would have silently reused the unpinned catalogue's cache.
    """
    base = dict(detection_window_h=12.0, detection_threshold=10.0,
                detection_centered=True, detection_peak_days=True)

    pinned = detection_tag(**base, time_step_h=FIVE_MIN, detection_step_h=1.0)
    unpinned = detection_tag(**base, time_step_h=FIVE_MIN, detection_step_h=None)
    assert pinned != unpinned

    # And it stays a no-op on hourly data, so the hourly caches and catalogues
    # are not invalidated by switching the pin on.
    assert (detection_tag(**base, time_step_h=HOURLY, detection_step_h=1.0)
            == detection_tag(**base, time_step_h=HOURLY, detection_step_h=None))


def test_get_events_can_express_every_setting_the_extraction_takes():
    """The defect in one line: get_events is the only way inference builds a
    catalogue, so a setting it cannot pass is a setting inference cannot
    match."""
    import inspect

    from swafi.utils.event_extraction import extract_events_parallel
    from swafi.utils.use_common import get_events

    extraction = inspect.signature(extract_events_parallel).parameters
    inference = set(inspect.signature(get_events).parameters)

    detection = {name for name in extraction if name.startswith('detection_')}
    assert detection <= inference, (
        f"get_events cannot express {sorted(detection - inference)}, so "
        f"inference cannot reproduce a training catalogue that uses it")


def test_no_inference_script_builds_its_own_event_definition():
    """The regression this file exists for.

    The settings belong to the catalogue, not to five scripts. Each copy is a
    copy that can drift, and a drifted copy is silent.
    """
    from pathlib import Path

    scripts = sorted((Path(__file__).parent.parent / 'scripts'
                      / 'impact_functions').glob('use_*.py'))
    assert scripts, "the inference scripts moved"

    for path in scripts:
        src = path.read_text(encoding='utf-8')
        if 'get_inference_events' not in src and 'get_events' not in src:
            continue  # a script that needs no events at all
        assert 'get_inference_events(' in src, (
            f"{path.name} builds its events outside the shared helper")
        assert 'DETECTION_WINDOW_H' not in src, (
            f"{path.name} keeps its own copy of the detection settings")
