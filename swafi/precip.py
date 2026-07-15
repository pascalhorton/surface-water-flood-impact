"""
Class to handle the precipitation forecast data.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import uniform_filter

from .config import Config
from .domain import Domain

config = Config()

# Temporal window for simple-event extraction (hours relative to D-day at 0 h)
SIMPLE_EVENT_HOURS_BEFORE = 8
SIMPLE_EVENT_HOURS_AFTER = 26


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

    def extract_events(self, coords_row=None, method='simple', simple_strict_mode=False, api_days_nb=30, api_reg=0.8):
        # Select timeseries and convert it into a DataFrame
        if coords_row is not None:
            return self._extract_events(coords_row, method, simple_strict_mode, api_days_nb, api_reg)

        list_of_events = []
        coords_df = self.domain.get_coordinates_df()
        for _, coords_row in tqdm(coords_df.iterrows(), total=len(coords_df), desc="Extracting events"):
            events = self._extract_events(coords_row, method, simple_strict_mode, api_days_nb, api_reg)
            if events is not None:
                list_of_events.append(events)

        return pd.concat(list_of_events, axis=0).reset_index(drop=True)

    def _extract_events(self, coords_row, method='simple', simple_strict_mode=False, api_days_nb=30, api_reg=0.8):
        time_series = self.data.sel(
            x=coords_row.x,
            y=coords_row.y
        ).to_dataframe().reset_index()

        # Time step [h] of the loaded data
        dt = (time_series['time'].iloc[1] - time_series['time'].iloc[0]).total_seconds() / 3600

        # Sub-hourly accumulation windows are only meaningful when the data
        # resolves them (e.g. 5-min data); skip windows shorter than the time step.
        window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]

        # Transform the precipitation data to percentiles
        time_series['precip_q'] = time_series['precip'].rank(pct=True)

        if method == 'classic':  # Bernet et al. (2019) method

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

            # q98 threshold on precipitation intensity
            threshold = time_series['precip'].quantile(0.98)

            # All exceedance timestamps
            exceed_times = time_series.loc[time_series['precip'] >= threshold, 'time']

            # Event dates
            events = self._build_simple_event_dates(exceed_times, simple_strict_mode)

            # Pre-compute rolling precipitation sums for each accumulation window
            window_hours = [1, 2, 4, 6, 12, 24, 48, 72]

            # Hour-based windows
            for W in window_hours:
                n_steps = max(1, int(round(W / dt)))
                time_series[f'p_{W}h'] = time_series['precip'].rolling(n_steps).sum()
                time_series[f'p_{W}h_q'] = time_series[f'p_{W}h'].rank(pct=True)
            
            # Minute-based windows
            for W in window_minutes:
                n_steps = max(1, int(round(W / 60 / dt)))
                time_series[f'p_{W}min'] = time_series['precip'].rolling(n_steps).sum()
                time_series[f'p_{W}min_q'] = time_series[f'p_{W}min'].rank(pct=True)

            # Get the date and time of the maximum precipitation intensity
            # Use searchsorted on a time-indexed series for O(log n) window lookup
            # instead of O(n) boolean masking, and collect results in a list to
            # avoid repeated pandas column reallocations inside the loop.
            ts_indexed = time_series.set_index('time')
            time_idx = ts_indexed.index
            records = []
            for _, row in events.iterrows():
                if simple_strict_mode:
                    start = row['e_date']
                    end = row['e_date'] + pd.Timedelta(hours=24)
                else:
                    start = row['e_date'] - pd.Timedelta(hours=SIMPLE_EVENT_HOURS_BEFORE)
                    end = row['e_date'] + pd.Timedelta(hours=SIMPLE_EVENT_HOURS_AFTER)
                i0 = time_idx.searchsorted(start, side='left')
                i1 = time_idx.searchsorted(end, side='right')
                window = ts_indexed.iloc[i0:i1]
                if window.empty:
                    rec = {'i_max': np.nan, 'i_max_q': np.nan, 'i_max_date': pd.NaT}
                    rec.update({f'p_{W}h': np.nan for W in window_hours})
                    rec.update({f'p_{W}h_q': np.nan for W in window_hours})
                    rec.update({f'p_{W}min': np.nan for W in window_minutes})
                    rec.update({f'p_{W}min_q': np.nan for W in window_minutes})
                else:
                    rec = {
                        # Intensity as mm/h regardless of the native time step
                        'i_max': window['precip'].max() / dt,
                        'i_max_q': window['precip_q'].max(),
                        'i_max_date': window['precip'].idxmax(),
                    }
                    for W in window_hours:
                        rec[f'p_{W}h'] = window[f'p_{W}h'].max()
                        rec[f'p_{W}h_q'] = window[f'p_{W}h_q'].max()
                    for W in window_minutes:
                        rec[f'p_{W}min'] = window[f'p_{W}min'].max()
                        rec[f'p_{W}min_q'] = window[f'p_{W}min_q'].max()
                records.append(rec)
            events = pd.concat(
                [events, pd.DataFrame(records, index=events.index)], axis=1
            )

            events = events.astype({
                **{f'p_{W}h': 'float32' for W in window_hours},
                **{f'p_{W}h_q': 'float32' for W in window_hours},
                **{f'p_{W}min': 'float32' for W in window_minutes},
                **{f'p_{W}min_q': 'float32' for W in window_minutes},
            })

            # Aggregate time series at daily time step
            daily_series = time_series.set_index('time').resample('D').agg({
                'precip': 'sum'
            })

            # Compute API on the daily series
            daily_series['api'] = self._compute_api(
                daily_series['precip'].values, 24, api_days_nb, api_reg
            )
            daily_series['api_q'] = daily_series['api'].rank(pct=True)

            # Attach API and its quantile to events
            events = events.merge(
                daily_series.reset_index()[['time', 'api', 'api_q']],
                left_on='e_date',
                right_on='time',
                how='left'
            ).drop(columns=['time'])

        else:
            raise ValueError(f"Unknown event extraction method: {method}")

        if len(events) == 0:
            return None

        # Add coordinates to the DataFrame and round all float values
        df_coords = pd.concat([pd.DataFrame(coords_row).T] * len(events),
                              ignore_index=True)
        events = pd.concat([df_coords, events], axis=1)
        float_cols = events.select_dtypes(include='float').columns
        events[float_cols] = events[float_cols].round(5)

        return events

    @staticmethod
    def _build_simple_event_dates(exceed_times, strict_mode):
        """Build unique event days from exceedance timestamps.

        Rules for non strict mode:
        - 00:00 <= t < 02:00 -> day itself and previous day
        - 02:00 <= t < 16:00 -> day itself
        - 16:00 <= t < 24:00 -> day itself and next day
        """
        day = exceed_times.dt.floor('D')
        if not strict_mode:
            time_of_day = exceed_times - day

            early_mask = time_of_day < pd.Timedelta(hours=2)
            late_mask = time_of_day >= pd.Timedelta(hours=16)

            candidate_days = pd.concat([
                day,
                day[early_mask] - pd.Timedelta(days=1),
                day[late_mask] + pd.Timedelta(days=1),
            ], ignore_index=True)
        else:
            candidate_days = day

        event_days = pd.Series(pd.to_datetime(candidate_days.to_numpy()))
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
