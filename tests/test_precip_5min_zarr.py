import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip('zarr')

from swafi.precip_combiprecip_5min import (
    GRID_RESOLUTION,
    GRID_X0,
    GRID_X_SIZE,
    GRID_Y0,
    GRID_Y_SIZE,
    RATE_TO_STEP,
    STEPS_PER_DAY,
    _init_zarr_template,
    _write_day_to_zarr,
)

DATE = pd.Timestamp('2021-06-01')  # DOY 152
Y_START, Y_END = 100, 132
X_START, X_END = 200, 248


def _make_day_zip(tmp_path, rates_by_member):
    """Create a daily CPCH zip with the given {member name: full-grid rates}."""
    zip_path = tmp_path / 'CPCHhdf521152.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for member, rates in rates_by_member.items():
            h5_path = tmp_path / member
            with h5py.File(h5_path, 'w') as h5:
                h5.create_dataset('dataset1/data1/data', data=rates)
            zf.write(h5_path, arcname=member)
    return zip_path


def test_zarr_store_day_write_and_nan_fill(tmp_path):
    rng = np.random.default_rng(7)
    rates = rng.uniform(0.0, 5.0, (GRID_Y_SIZE, GRID_X_SIZE)).astype('float32')
    rates[0, 0] = np.inf  # 'undetect' encoding -> 0 mm
    rates[110, 210] = np.inf

    # Two 5-min files (00:05 and 12:00); every other step of the day is missing
    zip_path = _make_day_zip(tmp_path, {
        'CPC2115200050_00005.801.h5': rates,
        'CPC2115212000_00005.801.h5': rates * 2.0,
    })

    # Two-day store cropped to a small window, as build_zarr_store() sets it up
    store = tmp_path / 'cpch_5min.zarr'
    time_coord = pd.date_range(DATE, periods=2 * STEPS_PER_DAY, freq='5min')
    x_axis = GRID_X0 + np.arange(GRID_X_SIZE) * GRID_RESOLUTION
    y_axis = GRID_Y0 - np.arange(GRID_Y_SIZE) * GRID_RESOLUTION
    _init_zarr_template(store, time_coord, y_axis[Y_START:Y_END],
                        x_axis[X_START:X_END], chunk_size=32)

    marker = tmp_path / 'done_2021-06-01'
    _write_day_to_zarr(str(store), str(zip_path), DATE, 0,
                       Y_START, Y_END, X_START, X_END, str(marker))
    assert marker.exists()

    data = xr.open_zarr(store, consolidated=False)['precip']

    # Written steps match the source rates converted to mm per 5-min step
    expected = np.where(np.isposinf(rates), 0.0, rates) * RATE_TO_STEP
    expected = expected[Y_START:Y_END, X_START:X_END].astype('float32')
    np.testing.assert_allclose(
        data.sel(time=DATE + pd.Timedelta(minutes=5)).values, expected, rtol=1e-6)
    np.testing.assert_allclose(
        data.sel(time=DATE + pd.Timedelta(hours=12)).values, 2 * expected, rtol=1e-6)
    assert float(data.sel(time=DATE + pd.Timedelta(hours=12),
                          y=y_axis[110], x=x_axis[210])) == 0.0  # undetect -> 0

    # Steps of the day without a source file are NaN
    assert np.isnan(data.sel(time=DATE + pd.Timedelta(minutes=10)).values).all()

    # The second day was never written: reads back as NaN (fill value)
    assert np.isnan(data.sel(time=slice('2021-06-02', None)).values).all()

    # Coordinates carry the cropped grid axes
    np.testing.assert_allclose(data['x'].values, x_axis[X_START:X_END])
    np.testing.assert_allclose(data['y'].values, y_axis[Y_START:Y_END])
