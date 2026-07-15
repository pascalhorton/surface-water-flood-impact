import pandas as pd

from swafi.precip import Precipitation


def test_build_simple_event_dates_respects_time_of_day_rules():
    exceed_times = pd.Series(
        pd.to_datetime(
            [
                '2020-01-10 00:00:00',
                '2020-01-10 01:59:00',
                '2020-01-10 02:00:00',
                '2020-01-10 15:59:00',
                '2020-01-10 16:00:00',
                '2020-01-10 23:59:00',
            ]
        )
    )

    events = Precipitation._build_simple_event_dates(exceed_times, strict_mode=False)

    expected = pd.to_datetime(
        [
            '2020-01-09 00:00:00',
            '2020-01-10 00:00:00',
            '2020-01-11 00:00:00',
        ]
    )
    pd.testing.assert_series_equal(events['e_date'], pd.Series(expected), check_names=False)


def test_build_simple_event_dates_deduplicates_and_sorts_days():
    exceed_times = pd.Series(
        pd.to_datetime(
            [
                '2020-01-11 18:00:00',
                '2020-01-10 01:00:00',
                '2020-01-10 01:30:00',
                '2020-01-11 03:00:00',
            ]
        )
    )

    events = Precipitation._build_simple_event_dates(exceed_times, strict_mode=False)

    expected = pd.to_datetime(
        [
            '2020-01-09 00:00:00',
            '2020-01-10 00:00:00',
            '2020-01-11 00:00:00',
            '2020-01-12 00:00:00',
        ]
    )
    pd.testing.assert_series_equal(events['e_date'], pd.Series(expected), check_names=False)

