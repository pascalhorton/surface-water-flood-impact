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

        self.data = self.data.fillna(0)
        self.data[self.precip_var].values = uniform_filter(
            self.data[self.precip_var],
            size=(0, filter_size, filter_size)
        )

    def extract_events(self, coords_row=None, method='simple', api_days_nb=30, api_reg=0.8):
        # Select timeseries and convert it into a DataFrame
        if coords_row is not None:
            return self._extract_events(coords_row, method, api_days_nb, api_reg)

        list_of_events = []
        coords_df = self.domain.get_coordinates_df()
        for _, coords_row in tqdm(coords_df.iterrows(), total=len(coords_df), desc="Extracting events"):
            events = self._extract_events(coords_row, method, api_days_nb, api_reg)
            if events is not None:
                list_of_events.append(events)

        return pd.concat(list_of_events, axis=0).reset_index(drop=True)

    def _extract_events(self, coords_row, method='simple', api_days_nb=30, api_reg=0.8):
        time_series = self.data.sel(
            x=coords_row.x,
            y=coords_row.y
        ).to_dataframe().reset_index()

        # Transform the precipitation data to percentiles
        time_series['precip_q'] = time_series['precip'].rank(pct=True)

        if method == 'classic':  # Bernet et al. (2019) method

            # Calculate the Antecedent Precipitation Index (API) using a convolution
            time_series["api"] = self.compute_api(
                time_series.precip.values, self.time_step, api_days_nb, api_reg)

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
            events = pd.concat([
                event_groups.time.agg(["first", "last", "size"]),
                event_groups.precip.agg(["sum", "max", "mean", "std"]),
                event_groups.api.first(),
                i_max_date.rename("i_max_date")
            ], axis=1)
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
            events = events.astype({
                "duration": "int16",
                "i_sd": "float32",
                "api": "float32"
            })

            # Drop events that do not fulfill the condition of minimal precipitation
            events = events[events.p_sum >= 10].reset_index(drop=True)

            # Calculate percentiles of score for each event characteristics
            ranks = events.iloc[:, 2:-1].rank(pct=True)
            ranks.columns = ["duration_q", "p_sum_q", "i_max_q", "i_mean_q", "i_sd_q",
                             "api_q"]
            events = pd.concat([events, ranks], axis=1)

        elif method == 'simple':  # New simple method based on the precipitation intensity

            # q98 threshold on precipitation intensity
            threshold = time_series['precip'].quantile(0.98)

            # All exceedance timestamps
            exceed_times = time_series.loc[time_series['precip'] >= threshold, 'time']

            # Event dates:
            # - day of exceedance
            # - day of exceedance shifted by +12h (captures days within 12h prior to start and 12h after start)
            events = pd.concat([
                exceed_times.dt.floor('D'),
                (exceed_times + pd.Timedelta(hours=-12)).dt.floor('D'),
                (exceed_times + pd.Timedelta(hours=12)).dt.floor('D')
            ]).drop_duplicates().sort_values().to_frame(name='e_date').reset_index(drop=True)

            # Pre-compute rolling precipitation sums for each accumulation window
            window_hours = [1, 2, 4, 6, 12, 24, 48, 72]
            dt = (time_series['time'].iloc[1] - time_series['time'].iloc[0]).total_seconds() / 3600
            for W in window_hours:
                n_steps = max(1, int(round(W / dt)))
                time_series[f'p_{W}h'] = time_series['precip'].rolling(n_steps).sum()
                time_series[f'p_{W}h_q'] = time_series[f'p_{W}h'].rank(pct=True)

            # Get the date and time of the maximum precipitation intensity
            for idx, row in events.iterrows():
                day_series = time_series[
                    (time_series['time'] >= row['e_date'] + pd.Timedelta(hours=-12)) &
                    (time_series['time'] <= row['e_date'] + pd.Timedelta(hours=36))
                ]
                i_max_date = day_series.loc[day_series['precip'].idxmax(), 'time']
                events.at[idx, 'i_max'] = day_series['precip'].max()
                events.at[idx, 'i_max_q'] = day_series['precip_q'].max()
                events.at[idx, 'i_max_date'] = i_max_date

                # Max rolling sum for each window, constrained to windows ending within
                # the day of i_max_date (right edge in (day_start, day_end])
                day_start = i_max_date.normalize()
                day_end = day_start + pd.Timedelta(hours=24)
                day_mask = (
                    (time_series['time'] > day_start) &
                    (time_series['time'] <= day_end)
                )
                for W in window_hours:
                    events.at[idx, f'p_{W}h'] = time_series.loc[day_mask, f'p_{W}h'].max()
                    events.at[idx, f'p_{W}h_q'] = time_series.loc[day_mask, f'p_{W}h_q'].max()

            events = events.astype({
                **{f'p_{W}h': 'float32' for W in window_hours},
                **{f'p_{W}h_q': 'float32' for W in window_hours},
            })

            # Aggregate time series at daily time step
            daily_series = time_series.set_index('time').resample('D').agg({
                'precip': 'sum'
            })

            # Compute API on the daily series
            daily_series['api'] = self.compute_api(
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

    def compute_api(self, precip, time_step, days_nb=30, reg=0.8):
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
