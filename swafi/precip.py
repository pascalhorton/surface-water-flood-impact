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
                       detection_step_h=None,
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
        detection_step_h: float|None
            Detect as if the data had this time step [h]: both the tested
            positions and the span of the detection window follow it. None
            (default) uses the native step, which ties the catalogue to the
            resolution of the data (see _extract_events_simple).
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
                                        detection_step_h,
                                        reference, collect_reference)

        list_of_events = []
        coords_df = self.domain.get_coordinates_df()
        for _, coords_row in tqdm(coords_df.iterrows(), total=len(coords_df), desc="Extracting events"):
            events = self._extract_events(coords_row, method, api_days_nb, api_reg,
                                          detection_window_h, detection_threshold,
                                          detection_centered, detection_peak_days,
                                          detection_step_h,
                                          reference, collect_reference)
            if events is not None:
                list_of_events.append(events)

        return pd.concat(list_of_events, axis=0).reset_index(drop=True)

    def _extract_events(self, coords_row, method='simple', api_days_nb=30, api_reg=0.8,
                        detection_window_h=1.0, detection_threshold=None,
                        detection_centered=True, detection_peak_days=True,
                        detection_step_h=None,
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
                detection_centered, detection_peak_days, detection_step_h,
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

    @staticmethod
    def _detection_span(detection_window_h, dt, detection_centered=True,
                        detection_step_h=None):
        """Inclusive step offsets of the detection window around its label.

        The window is a number of steps, so its extent in wall-clock time
        follows the resolution of the data unless something pins it. With
        end-labelled values (the step labelled T holds the accumulation over
        (T - dt, T]) and pandas' centring, a 12-hour window covers

            (t - 7h,    t + 5h   ]   on hourly data,      labels [t-6, t+5]
            (t - 6h05,  t + 5h55 ]   at five minutes,     labels [t-72, t+71]

        the same twelve hours offset by one coarse step. Two catalogues can
        therefore test the very same instants and still disagree on what the
        sum there is.

        detection_step_h pins it: the window is laid out as a product of that
        step would lay it out, then converted to native steps, so the covered
        interval is the same whatever the data resolution. None keeps the
        native layout, which is the behaviour of every extraction predating
        this.

        Returns
        -------
        tuple(int, int)
            (lo, hi): the window at label i covers precip[i + lo : i + hi + 1].
        """
        if detection_window_h is None:
            return 0, 0

        ref = dt if detection_step_h is None else float(detection_step_h)
        w_ref = max(1, int(round(detection_window_h / ref)))
        per_ref = ref / dt  # native steps in one reference step

        if detection_centered:
            # pandas' centring for a window of w_ref reference steps puts the
            # label at [t - w_ref // 2, t + w_ref - 1 - w_ref // 2]; with
            # end-labelling that is the interval (t - (k_lo + 1) * ref,
            # t + k_hi * ref].
            k_lo = w_ref // 2
            k_hi = w_ref - 1 - k_lo
        else:
            # Trailing: (t - w_ref * ref, t].
            k_lo, k_hi = w_ref - 1, 0

        lo = -int(round((k_lo + 1) * per_ref)) + 1
        hi = int(round(k_hi * per_ref))
        return lo, hi

    @staticmethod
    def _detection_series(precip, lo_off, hi_off):
        """Sum of precip over [i + lo_off, i + hi_off] at every label i.

        Bit-identical to rolling(w, center=True).sum() when the offsets are
        the ones pandas' centring produces, which a test pins: pandas builds
        a centred window the same way, by shifting a trailing one.
        """
        n = hi_off - lo_off + 1
        if n == 1 and hi_off == 0:
            return precip  # a one-step window keeps the native values
        rolled = pd.Series(precip).rolling(n).sum().to_numpy()
        if hi_off == 0:
            return rolled
        out = np.full(rolled.shape, np.nan)
        if hi_off > 0:
            out[:-hi_off] = rolled[hi_off:]
        else:
            out[-hi_off:] = rolled[:hi_off]
        return out

    @staticmethod
    def _detection_positions(times, dt, detection_step_h=None):
        """Boolean mask of the steps the detection threshold is tested on.

        None tests every native step, which is what ties a catalogue to the
        resolution it was detected on. A reference step in hours tests only
        the timestamps on that wall-clock grid, anchored at midnight, so an
        hourly reference picks out of 5-minute data the same instants an
        hourly series offers of its own accord.

        Parameters
        ----------
        times: pd.DatetimeIndex
            The time axis of the cell.
        dt: float
            The time step [h] of that axis.
        detection_step_h: float|None
            The wall-clock spacing [h] between tested positions.

        Returns
        -------
        np.ndarray
            A boolean mask over times.
        """
        if detection_step_h is None:
            return np.ones(len(times), dtype=bool)

        stride_min = int(round(detection_step_h * 60))
        step_min = int(round(dt * 60))
        if stride_min <= 0:
            raise ValueError(
                f"detection_step_h must be positive, got "
                f"{detection_step_h}.")
        if step_min <= 0:
            raise ValueError(f"The time step is {dt} h, which rounds to 0 min.")
        if stride_min % step_min != 0:
            raise ValueError(
                f"detection_step_h={detection_step_h} is {stride_min} min, "
                f"which is not a multiple of the {step_min} min time step: "
                f"most of the grid it asks for does not exist in the data.")
        if 1440 % stride_min != 0:
            raise ValueError(
                f"detection_step_h={detection_step_h} is {stride_min} min, "
                f"which does not divide a day: the grid would shift from one "
                f"day to the next and the catalogue would depend on where the "
                f"series happens to start.")
        if stride_min == step_min:
            # Every step is on the grid; keep it an explicit no-op so that the
            # file-name tag and the behaviour agree.
            return np.ones(len(times), dtype=bool)

        minute_of_day = (times.hour * 60 + times.minute).to_numpy()
        return ((minute_of_day % stride_min == 0)
                & (times.second.to_numpy() == 0)
                & (times.microsecond.to_numpy() == 0))

    @staticmethod
    def _peak_search_series(precip_filled, eval_mask, dt,
                            detection_step_h=None):
        """The series the event date is read off (see _build_peak_event_dates).

        The date comes from the intensity peak inside the exceeding window,
        and on a 5-minute product the heaviest five minutes and the heaviest
        hour are different instants: either side of midnight they date the
        same storm on different days. That precision is never used - the peak
        is floored to a day and the event becomes [e_date, e_date + 24h] - so
        searching it natively adds no information, only a date that moves with
        the resolution of the product, and a date that moves changes which
        claims link to the event.

        detection_step_h therefore governs the search too: the candidates are
        the accumulations over one reference step, taken at the reference
        positions, which is exactly what a product of that step would offer.
        An hourly reference on 5-minute data searches the same 24 hourly
        totals an hourly series holds, so the date is the one the hourly
        extraction gives.

        None, or a reference step equal to the native one, returns the input
        untouched.

        Parameters
        ----------
        precip_filled: np.ndarray
            The native precipitation with the invalid steps at -inf.
        eval_mask: np.ndarray
            The boolean mask of the reference positions (_detection_positions).
        dt: float
            The native time step [h].
        detection_step_h: float|None
            The reference step [h] the detection is pinned to.

        Returns
        -------
        np.ndarray
            The series to search, -inf wherever no candidate sits.
        """
        if detection_step_h is None:
            return precip_filled
        per_ref = int(round(float(detection_step_h) / dt))
        if per_ref <= 1:
            return precip_filled
        # Trailing, like the native values themselves: the reference step
        # labelled T holds (T - detection_step_h, T]. A gap anywhere inside it
        # carries the -inf out, so an incomplete reference step can never be
        # taken for a peak - the same thing an hourly product does by leaving
        # the hour missing.
        totals = pd.Series(precip_filled).rolling(per_ref).sum().to_numpy()
        return np.where(eval_mask & np.isfinite(totals), totals, -np.inf)

    def _extract_events_simple(self, times, precip, dt, window_minutes,
                               api_days_nb, api_reg, detection_window_h=1.0,
                               detection_threshold=None, detection_centered=True,
                               detection_peak_days=True, detection_step_h=None,
                               ref=None, build_ref=False):
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
        detection_step_h: float|None
            Detect as if the data had this time step [h], anchored at
            midnight. None (default) uses the native step.

            WHY THIS EXISTS. Two things in the detection follow the resolution
            of the data rather than the weather.

            The POSITIONS: the window slides with the time step, so the number
            of chances a day gets to cross the threshold scales with the
            resolution - 24 a day at an hourly step, 288 at five minutes. A
            day whose 12-hour sum peaks at 9.8 mm over hourly offsets can
            reach 10 mm at some 5-minute offset, so it enters one catalogue
            and not the other. Measured in Phase Q: the hourly and 5-minute
            catalogues each held 91.0% of the 3,024 damaging days in their
            union, and the 9% they disagreed on were borderline crossings of
            this kind, not differences in the rain.

            The WINDOW: a window of w steps covers w * dt of wall clock, and
            with end-labelling and pandas' centring a 12-hour window sits at
            (t-7h, t+5h] hourly against (t-6h05, t+5h55] at five minutes. Two
            catalogues can test the same instant and still disagree on the sum
            there.

            The DATE: an event is dated on the intensity peak inside its
            exceeding window, and that peak was looked for on the native
            series, so the heaviest five minutes and the heaviest hour date
            the same storm on different days whenever they straddle midnight.
            The precision is never used downstream (the peak is floored to a
            day), so it bought nothing and moved dates, and a moved date
            changes which claims link to the event.

            This parameter pins all three, so a 5-minute extraction with
            detection_step_h=1.0 asks the same question of the same instants
            as an hourly extraction does and dates the answer the same way;
            the two catalogues then differ only where the underlying products
            differ. It is exactly a no-op on hourly data, which a test pins,
            so the hourly catalogue does not move when this is switched on.
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
        # Span of the detection window relative to the step it labels, and
        # the accumulation over it. Both follow detection_step_h, so that a
        # finer product sums the same wall-clock interval as a coarse one.
        lo_off, hi_off = self._detection_span(
            detection_window_h, dt, detection_centered, detection_step_h)
        detection = self._detection_series(precip, lo_off, hi_off)
        detection_valid = np.isfinite(detection)
        if not detection_valid.any():
            return None, None
        # The positions the threshold is tested on (see detection_step_h).
        eval_mask = self._detection_positions(times, dt, detection_step_h)
        detection_valid &= eval_mask
        if not detection_valid.any():
            return None, None
        if detection_threshold is not None:
            # Already period-independent: the reference does not apply here.
            threshold = float(detection_threshold)
        elif ref is not None:
            threshold = ref['q98']
        else:
            # Over the tested positions only, for the same reason
            # detection_step_h exists: sampling one rolling window more often
            # oversamples its peaks, so the q98 of a 12-hour sum read every
            # 5 minutes is not the q98 of the same sum read hourly.
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

        exceed = (detection >= threshold) & eval_mask
        if detection_peak_days:
            # The peak is searched on the reference step the detection is
            # pinned to, so that the date is a property of the weather and not
            # of the product's sampling rate (see _peak_search_series).
            events = self._build_peak_event_dates(
                times,
                self._peak_search_series(precip_filled, eval_mask, dt,
                                         detection_step_h),
                exceed, lo_off, hi_off)
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
            The series to read the peak off, with the positions that hold no
            candidate at -inf. _peak_search_series builds it: the native
            values when the detection is not pinned, the reference-step
            accumulations at the reference positions when it is.
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
