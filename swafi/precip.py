"""
Class to handle the precipitation forecast data.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve

from .config import Config
from .domain import Domain
from .utils.precip_reference import build_cdf, cdf_percentile

config = Config()


class Precipitation:
    def __init__(self, cid_file=None):
        """
        The generic PrecipitationForecast class.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        """
        if not cid_file:
            cid_file = config.get('CID_PATH', None, False)

        self.data = None
        self.dataset_name = None
        self.data_path = None
        self.x_axis_dim = 'x'
        self.y_axis_dim = 'y'
        self.time_axis_dim = 'time'
        self.precip_var = 'precip'

        self.domain = Domain(cid_file)
        self.resolution = None
        self.time_step = None
        self.tmp_dir = Path(config.get('TMP_DIR'))

    def set_data_path(self, data_path):
        """
        Set the path to the precipitation data.

        Parameters
        ----------
        data_path: str
            The path to the precipitation data
        """
        self.data_path = data_path

    def prepare_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def apply_smoothing(self, filter_size=3):
        """
        Apply a uniform filter to the precipitation data.

        Parameters
        ----------
        filter_size: int
            The size of the filter (default: 3)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load the data first.")

        if filter_size is None:
            return

        self.data = self.data.fillna(0)
        self.data[self.precip_var].values = uniform_filter(
            self.data[self.precip_var],
            size=(0, filter_size, filter_size)
        )

    def extract_events(self, coords_row=None, method='simple', api_days_nb=30, api_reg=0.8,
                       detection_window_h=1.0, detection_threshold=None,
                       detection_centered=True, detection_peak_days=True,
                       reference=None, collect_reference=None):
        """
        Extract events for one cell (coords_row given) or the whole domain.

        detection_threshold: float|None
            An absolute detection threshold [mm] on the detection window
            accumulation (e.g. 10 with detection_window_h=12 for p_12h >= 10mm).
            None uses the per-cell q98 of that accumulation (default).
        detection_centered: bool
            Whether the detection window is centred on the step it labels
            instead of trailing it (a no-op for a 1-step window).
        detection_peak_days: bool
            Whether to date the events on the intensity peak of each exceeding
            window instead of on the exceedances themselves.
        reference: dict|None
            Mapping cid -> per-cell training reference to normalise against
            (see _extract_events_simple). None re-estimates on the current period.
        collect_reference: dict|None
            When given, the computed per-cell reference is written into it
            (keyed by cid) so the training normalisation can be persisted.
        """
        # Select timeseries and convert it into a DataFrame
        if coords_row is not None:
            return self._extract_events(coords_row, method, api_days_nb, api_reg,
                                        detection_window_h, detection_threshold,
                                        detection_centered, detection_peak_days,
                                        reference, collect_reference)

        list_of_events = []
        coords_df = self.domain.get_coordinates_df()
        for _, coords_row in tqdm(coords_df.iterrows(), total=len(coords_df), desc="Extracting events"):
            events = self._extract_events(coords_row, method, api_days_nb, api_reg,
                                          detection_window_h, detection_threshold,
                                          detection_centered, detection_peak_days,
                                          reference, collect_reference)
            if events is not None:
                list_of_events.append(events)

        return pd.concat(list_of_events, axis=0).reset_index(drop=True)

    def _extract_events(self, coords_row, method='simple', api_days_nb=30, api_reg=0.8,
                        detection_window_h=1.0, detection_threshold=None,
                        detection_centered=True, detection_peak_days=True,
                        reference=None, collect_reference=None):
        cell = self.data.sel(x=coords_row.x, y=coords_row.y)
        times = pd.DatetimeIndex(pd.to_datetime(cell['time'].values))
        precip = np.asarray(cell['precip'].values, dtype='float64').reshape(-1)

        # Time step [h] of the loaded data
        dt = (times[1] - times[0]).total_seconds() / 3600

        # Sub-hourly accumulation windows are only meaningful when the data
        # resolves them (e.g. 5-min data); skip windows shorter than the time step.
        window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]

        if method == 'classic':  # Bernet et al. (2019) method
            time_series = pd.DataFrame({'time': times, 'precip': precip})

            # Calculate the Antecedent Precipitation Index (API) using a convolution
            time_series["api"] = self._compute_api(
                time_series.precip.values, self.time_step, api_days_nb, api_reg)

            # Pre-compute rolling precipitation sums for short-duration windows
            for W in window_minutes:
                n_steps = max(1, int(round(W / 60 / dt)))
                time_series[f'p_{W}min'] = time_series['precip'].rolling(n_steps).sum()

            # Group events by period of at least 8 hour without precipitation larger than 0.1mm/h and return group IDs
            time_series_th = time_series[time_series.precip >= 0.1]
            group_ids = time_series_th.groupby(
                time_series_th.time.diff().gt("8h").cumsum()).ngroup() + 1

            # Fill gaps between events to correctly calculate all event characteristics and then group again
            time_series["group_ID"] = 0
            time_series.group_ID = group_ids
            ff = time_series.group_ID.ffill()
            bf = time_series.group_ID.bfill()
            time_series.group_ID = ff[ff == bf]
            event_groups = time_series.groupby("group_ID")

            # Get the date and time of the maximum precipitation intensity
            i_max_date = event_groups.apply(
                lambda g: g.loc[g.precip.idxmax(), 'time'],
                include_groups=False
            )

            # Calculate all precipitation characteristics
            short_window_cols = [f'p_{W}min' for W in window_minutes]
            short_window_aggs = {col: 'max' for col in short_window_cols}

            pieces = [
                event_groups.time.agg(["first", "last", "size"]),
                event_groups.precip.agg(["sum", "max", "mean", "std"]),
            ]
            if short_window_cols:
                pieces.append(event_groups[short_window_cols].agg(short_window_aggs))
            pieces += [
                event_groups.api.first(),
                i_max_date.rename("i_max_date")
            ]
            events = pd.concat(pieces, axis=1)
            events = events.rename(columns={
                "first": "e_start",
                "last": "e_end",
                "size": "duration",
                "sum": "p_sum",
                "max": "i_max",
                "mean": "i_mean",
                "std": "i_sd",
                "api": "api",
                "i_max_date": "i_max_date"
            })

            # Express intensities as mm/h regardless of the native time step
            # (the source values are accumulations per step, i.e. mm/step)
            events[["i_max", "i_mean", "i_sd"]] /= dt

            events = events.astype({
                "duration": "int16",
                "i_sd": "float32",
                "api": "float32",
                **{f'p_{W}min': 'float32' for W in window_minutes},
            })

            # Drop events that do not fulfill the condition of minimal precipitation
            events = events[events.p_sum >= 10].reset_index(drop=True)

            # Calculate percentiles of score for each event characteristics
            # Include short-duration windows in the quantile computation
            quantile_cols = ["duration", "p_sum", "i_max", "i_mean", "i_sd", "api"] + [f'p_{W}min' for W in window_minutes]
            ranks = events[quantile_cols].rank(pct=True)
            ranks.columns = ["duration_q", "p_sum_q", "i_max_q", "i_mean_q", "i_sd_q", "api_q"] + [f'p_{W}min_q' for W in window_minutes]
            events = pd.concat([events, ranks], axis=1)

        elif method == 'simple':  # New simple method based on the precipitation intensity
            cid = coords_row['cid']
            ref_in = reference.get(cid) if reference is not None else None
            events, cell_ref = self._extract_events_simple(
                times, precip, dt, window_minutes,
                api_days_nb, api_reg, detection_window_h, detection_threshold,
                detection_centered, detection_peak_days,
                ref=ref_in, build_ref=collect_reference is not None)
            if collect_reference is not None and cell_ref is not None:
                collect_reference[cid] = cell_ref

        else:
            raise ValueError(f"Unknown event extraction method: {method}")

        if events is None or len(events) == 0:
            return None

        # Add coordinates to the DataFrame and round all float values
        df_coords = pd.concat([pd.DataFrame(coords_row).T] * len(events),
                              ignore_index=True)
        events = pd.concat([df_coords, events], axis=1)
        float_cols = events.select_dtypes(include='float').columns
        events[float_cols] = events[float_cols].round(5)

        return events

    def _extract_events_simple(self, times, precip, dt, window_minutes,
                               api_days_nb, api_reg, detection_window_h=1.0,
                               detection_threshold=None, detection_centered=True,
                               detection_peak_days=True, ref=None, build_ref=False):
        """
        Simple event extraction on numpy arrays. Reproduces the per-window
        pandas rolling/rank/max results, but takes the per-event maxima on
        plain array slices and looks quantiles up only at the event maxima
        (rank is monotonic, so the max of the ranks over a window is the rank
        of the window maximum) instead of ranking the full columns.

        Parameters
        ----------
        times: pd.DatetimeIndex
            The (complete, sorted) time axis of the cell.
        precip: np.ndarray
            The precipitation values [mm/step] (float64, may contain NaN).
        dt: float
            The time step [h].
        window_minutes: list
            The sub-hourly accumulation windows [min] to compute.
        api_days_nb: int
            The number of days for the API calculation.
        api_reg: float
            The API recession constant.
        detection_window_h: float|None
            The accumulation window [h] on which the detection threshold
            is applied (default: 1 h). None means the native time step, i.e.
            for the 5-min dataset the events are detected on the 5-min bursts
            instead of the rolling hourly intensity.
        detection_threshold: float|None
            An absolute detection threshold [mm] on that accumulation, e.g.
            10 with detection_window_h=12 selects the days where p_12h reaches
            10 mm. None (default) uses the per-cell q98 of the accumulation,
            i.e. a relative, per-cell threshold. An absolute threshold is
            period- and cell-independent by construction, so ``ref`` is then
            not used for the detection (it still drives the ``*_q`` features).
        detection_centered: bool
            Whether the detection window is centred on the step it labels
            instead of trailing it. A no-op for a 1-step window (the default),
            where the two coincide.
        detection_peak_days: bool
            Whether to date the events on the intensity peak of each exceeding
            window (see _build_peak_event_dates) instead of on the exceedances
            themselves. False (default) keeps one event per exceedance day,
            which counts a storm once per day its detection window slides over.
        ref: dict|None
            A per-cell training reference (q98 threshold + CDFs) to normalise
            against, produced by a previous extraction with build_ref=True. When
            given, the detection threshold and every ``*_q`` are computed against
            the reference distribution instead of the current period, keeping the
            feature space consistent across the train/test boundary. When None,
            the normalisation is estimated on the current period (unchanged
            behaviour).
        build_ref: bool
            Whether to also compute and return this cell's reference (the q98
            threshold and the CDFs of raw precip, each accumulation window, and
            the daily API). Used when extracting the training events.

        Returns
        -------
        tuple(pd.DataFrame|None, dict|None)
            The events with their characteristics (or None if no valid data),
            and the cell reference (or None when build_ref is False).
        """
        valid_mask = ~np.isnan(precip)
        if not valid_mask.any():
            return None, None

        ref_out = {'windows': {}} if build_ref else None

        # Detection threshold on the precipitation intensity accumulated over
        # the detection window (a 1-step window keeps the native values
        # untouched): either an absolute value [mm] or the per-cell q98 of that
        # accumulation. The window is right-labelled like the p_*h columns,
        # unless centred: a centred window flags the steps the rain falls
        # around, a trailing one flags the steps it accumulated before, which
        # lags the storm by up to the window length.
        if detection_window_h is None:
            w_det = 1
        else:
            w_det = max(1, int(round(detection_window_h / dt)))
        if w_det == 1:
            detection = precip
        elif detection_centered:
            detection = pd.Series(precip).rolling(
                w_det, center=True).sum().to_numpy()
        else:
            detection = pd.Series(precip).rolling(w_det).sum().to_numpy()
        # Span of the detection window relative to the step it labels
        if detection_centered:
            lo_off, hi_off = -(w_det // 2), w_det - 1 - w_det // 2
        else:
            lo_off, hi_off = -(w_det - 1), 0
        detection_valid = np.isfinite(detection)
        if not detection_valid.any():
            return None, None
        if detection_threshold is not None:
            # Already period-independent: the reference does not apply here.
            threshold = float(detection_threshold)
        elif ref is not None:
            threshold = ref['q98']
        else:
            threshold = np.quantile(detection[detection_valid], 0.98)
        if build_ref:
            # The detection threshold actually applied on this cell (the q98 of
            # the detection window unless an absolute threshold was given).
            ref_out['q98'] = float(threshold)

        window_hours = [1, 2, 4, 6, 12, 24, 48, 72]
        window_defs = [(f'p_{W}h', max(1, int(round(W / dt))))
                       for W in window_hours]
        window_defs += [(f'p_{W}min', max(1, int(round(W / 60 / dt))))
                        for W in window_minutes]

        # Rolling sums per window (pandas, bit-identical to the previous
        # implementation — a shared cumulative sum would reorder the float
        # additions and perturb rank ties). -inf marks invalid steps
        # (incomplete window or NaN inside the window) so that a plain max()
        # skips them without NaN handling.
        n = precip.size
        m = len(window_defs)
        precip_series = pd.Series(precip)
        win_sums = np.empty((n, m))
        for k, (_, w) in enumerate(window_defs):
            col = precip_series.rolling(w).sum().to_numpy()
            win_sums[:, k] = np.where(np.isnan(col), -np.inf, col)
        precip_filled = np.where(valid_mask, precip, -np.inf)

        # Record the per-column CDFs of the current (training) period so that a
        # later extraction can rank its events against this same distribution.
        if build_ref:
            ref_out['precip'] = build_cdf(precip[valid_mask])
            for k, (name, _) in enumerate(window_defs):
                ref_out['windows'][name] = build_cdf(
                    win_sums[:, k][np.isfinite(win_sums[:, k])])

        exceed = detection >= threshold
        if detection_peak_days:
            events = self._build_peak_event_dates(
                times, precip_filled, exceed, lo_off, hi_off)
        else:
            events = self._build_simple_event_dates(pd.Series(times[exceed]))
        if len(events) == 0:
            # Still return the reference (built from the full series, not events)
            # so that cells without training events are covered at test time.
            if build_ref:
                ref_out['api'] = self._build_api_reference(
                    times, precip, api_days_nb, api_reg)
            return None, ref_out

        # Event windows (inclusive bounds, like searchsorted left/right)
        e_dates = pd.DatetimeIndex(events['e_date'])
        starts = e_dates
        ends = e_dates + pd.Timedelta(hours=24)
        i0 = times.searchsorted(starts, side='left')
        i1 = times.searchsorted(ends, side='right')

        # Per-event maxima over the window (one vectorized call per event)
        n_ev = len(events)
        times_arr = times.values
        p_max = np.full((n_ev, m), -np.inf)
        v_max = np.full(n_ev, -np.inf)
        i_max_date = np.full(n_ev, np.datetime64('NaT'), dtype='datetime64[ns]')
        for k in range(n_ev):
            a, b = i0[k], i1[k]
            if b <= a:
                continue  # event day outside the data period
            p_max[k] = win_sums[a:b].max(axis=0)
            block = precip_filled[a:b]
            j = int(block.argmax())
            if np.isneginf(block[j]):
                continue  # no valid precipitation in the window
            v_max[k] = block[j]
            i_max_date[k] = times_arr[a + j]

        # Quantiles of the maxima: against the training reference when given,
        # otherwise within the full series of each column (current period).
        i_max = np.where(np.isneginf(v_max), np.nan, v_max / dt)
        v_max_q = np.where(np.isneginf(v_max), np.nan, v_max)
        if ref is not None:
            i_max_q = cdf_percentile(ref.get('precip'), v_max_q)
        else:
            sorted_precip = np.sort(precip[valid_mask])
            i_max_q = self._pct_rank(sorted_precip, v_max)
        data = {
            # Intensity as mm/h regardless of the native time step
            'i_max': i_max,
            'i_max_q': i_max_q,
            'i_max_date': i_max_date,
        }
        for k, (name, _) in enumerate(window_defs):
            col = win_sums[:, k]
            p_max_col = np.where(np.isneginf(p_max[:, k]), np.nan, p_max[:, k])
            data[name] = p_max_col.astype('float32')
            if ref is not None:
                data[f'{name}_q'] = cdf_percentile(
                    ref['windows'].get(name), p_max_col).astype('float32')
            else:
                sorted_col = np.sort(col[np.isfinite(col)])
                data[f'{name}_q'] = self._pct_rank(
                    sorted_col, p_max[:, k]).astype('float32')
        events = pd.concat(
            [events, pd.DataFrame(data, index=events.index)], axis=1)

        # Aggregate time series at daily time step
        daily_series = pd.DataFrame(
            {'precip': precip},
            index=pd.DatetimeIndex(times, name='time')
        ).resample('D').agg({'precip': 'sum'})

        # Compute API on the daily series
        daily_series['api'] = self._compute_api(
            daily_series['precip'].values, 24, api_days_nb, api_reg
        )
        if build_ref:
            ref_out['api'] = build_cdf(daily_series['api'].values)

        if ref is not None:
            # Attach the API value, then rank it against the training reference.
            events = events.merge(
                daily_series.reset_index()[['time', 'api']],
                left_on='e_date', right_on='time', how='left').drop(columns=['time'])
            events['api_q'] = cdf_percentile(
                ref.get('api'), events['api'].to_numpy()).astype('float32')
        else:
            daily_series['api_q'] = daily_series['api'].rank(pct=True)
            events = events.merge(
                daily_series.reset_index()[['time', 'api', 'api_q']],
                left_on='e_date', right_on='time', how='left').drop(columns=['time'])

        return events, ref_out

    def _build_api_reference(self, times, precip, api_days_nb, api_reg):
        """
        Build the CDF of the daily API for a cell (used as the reference for
        api_q), for the case where the cell has no training events but must
        still carry a reference.
        """
        daily_series = pd.DataFrame(
            {'precip': precip},
            index=pd.DatetimeIndex(times, name='time')
        ).resample('D').agg({'precip': 'sum'})
        api = self._compute_api(
            daily_series['precip'].values, 24, api_days_nb, api_reg)
        return build_cdf(api)

    @staticmethod
    def _pct_rank(sorted_vals, values):
        """
        Percentile of each value within sorted_vals, matching pandas
        rank(pct=True) with average tie-handling. Non-finite values map to NaN.

        Parameters
        ----------
        sorted_vals: np.ndarray
            The sorted, finite sample the percentiles refer to.
        values: np.ndarray
            The values to rank.

        Returns
        -------
        np.ndarray
            The percentiles (NaN where the input is not finite).
        """
        values = np.asarray(values, dtype='float64')
        out = np.full(values.shape, np.nan)
        ok = np.isfinite(values)
        if sorted_vals.size and ok.any():
            left = np.searchsorted(sorted_vals, values[ok], side='left')
            right = np.searchsorted(sorted_vals, values[ok], side='right')
            out[ok] = (left + right + 1) / 2.0 / sorted_vals.size
        return out

    @staticmethod
    def _build_simple_event_dates(exceed_times):
        """Build unique, sorted event days from exceedance timestamps."""
        candidate_days = exceed_times.dt.floor('D')

        event_days = pd.Series(pd.to_datetime(candidate_days.to_numpy()))
        event_days = event_days.drop_duplicates().sort_values().reset_index(drop=True)

        return event_days.to_frame(name='e_date')

    @staticmethod
    def _build_peak_event_dates(times, precip_filled, exceed, lo_off, hi_off):
        """Date the events on the intensity peak of each exceeding window.

        Dating an event on the exceedance itself counts one storm several
        times: the detection window keeps the accumulation above the threshold
        while it slides over the storm, so a run of steps is flagged and it
        readily straddles midnight. Every one of those windows is however the
        same storm, and points at the same intensity peak: taking the day of
        that peak instead collapses them to one event.

        The rule adapts to what the window actually covers, without any
        merging heuristic: windows over a single storm share its peak and give
        one day, whereas a long spell whose windows peak on different days
        keeps one event per distinct peak day.

        Parameters
        ----------
        times: pd.DatetimeIndex
            The time axis of the cell.
        precip_filled: np.ndarray
            The precipitation values with NaN replaced by -inf.
        exceed: np.ndarray
            The boolean mask of the detection exceedances.
        lo_off, hi_off: int
            The span of the detection window relative to the step it labels
            (e.g. -6 and +5 for a centred 12-step window).

        Returns
        -------
        pd.DataFrame
            The unique, sorted event days.
        """
        idx = np.flatnonzero(exceed)
        if idx.size == 0:
            return pd.DataFrame({'e_date': pd.DatetimeIndex([])})

        n = precip_filled.size
        offsets = np.arange(lo_off, hi_off + 1)
        peaks = np.empty(idx.size, dtype='int64')
        # Chunked so that the (n_exceed x window) lookup stays bounded: at the
        # 5-min resolution a long window over a whole period would otherwise
        # materialise hundreds of MB.
        chunk = max(1, 2_000_000 // offsets.size)
        for s in range(0, idx.size, chunk):
            sel = idx[s:s + chunk]
            # Clipping repeats the edge value, which cannot beat the true
            # maximum of the window and maps back to the edge index anyway.
            pos = np.clip(sel[:, None] + offsets[None, :], 0, n - 1)
            j = precip_filled[pos].argmax(axis=1)
            peaks[s:s + sel.size] = pos[np.arange(sel.size), j]

        peaks = np.unique(peaks)
        event_days = pd.Series(times[peaks]).dt.floor('D')
        event_days = event_days.drop_duplicates().sort_values().reset_index(drop=True)

        return event_days.to_frame(name='e_date')

    @staticmethod
    def _compute_api(precip, time_step, days_nb=30, reg=0.8):
        """
        Compute the Antecedent Precipitation Index (API) for a given time series.

        Parameters
        ----------
        precip: np.ndarray
            The precipitation time series.
        time_step: float
            The time step of the precipitation data (in hours).
        days_nb: int
            The number of days to consider for the API calculation.
        reg: float
            The recession constant (between 0 and 1).

        Returns
        -------
        pd.Series
            The computed API values.
        """
        ts_per_day = 24 / time_step
        window = days_nb * ts_per_day
        kernel = np.power(reg, np.arange(window) / ts_per_day)
        # FFT convolution for long series (e.g. classic method on 5-min data,
        # where the direct product is ~1e9 operations per cell). NaNs need the
        # direct method: FFT would smear them over the whole series.
        if precip.size * kernel.size > 1e7 and not np.isnan(precip).any():
            api_full = fftconvolve(precip, kernel, mode="full")
        else:
            api_full = np.convolve(precip, kernel, mode="full")

        return np.concatenate(([0.0], api_full[:len(precip) - 1]))

    def get_x_axis_for_bounds(self, x_min, x_max):
        """
        Get the x-axis slice for the given bounds.

        Parameters
        ----------
        x_min: float
            The minimum x coordinate
        x_max: float
            The maximum x coordinate

        Returns
        -------
        slice
            The slice for the x-axis
        """
        x_axis = self.domain.cids['xs'][0, :]

        return x_axis[(x_axis >= x_min) & (x_axis <= x_max)]

    def get_y_axis_for_bounds(self, y_min, y_max):
        """
        Get the y-axis slice for the given bounds.

        Parameters
        ----------
        y_min: float
            The minimum y coordinate
        y_max: float
            The maximum y coordinate

        Returns
        -------
        slice
            The slice for the y-axis
        """
        y_axis = self.domain.cids['ys'][:, 0]

        return y_axis[(y_axis >= y_min) & (y_axis <= y_max)]
