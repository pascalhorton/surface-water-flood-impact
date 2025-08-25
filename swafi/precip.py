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
        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'time'
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

    def extract_events(self, coords_row=None, api_days_nb=30, api_reg=0.8):
        # Select timeseries and convert it into a DataFrame
        if coords_row is not None:
            return self._extract_events(coords_row, api_days_nb, api_reg)

        list_of_events = []
        coords_df = self.domain.get_coordinates_df()
        for _, coords_row in tqdm(coords_df.iterrows(), total=len(coords_df), desc="Extracting events"):
            events = self._extract_events(coords_row, api_days_nb, api_reg)
            if events is not None:
                list_of_events.append(events)

        return pd.concat(list_of_events, axis=0).reset_index(drop=True)

    def _extract_events(self, coords_row, api_days_nb=30, api_reg=0.8):
        time_series = self.data.sel(
            x=coords_row.x,
            y=coords_row.y
        ).to_dataframe().reset_index()

        # Calculate the Antecedent Precipitation Index (API) using a convolution
        time_series["api"] = self.compute_api(
            time_series.precip.values, api_days_nb, api_reg)

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

        if len(events) == 0:
            return None

        # Add coordinates to the DataFrame and round all float values
        df_coords = pd.concat([pd.DataFrame(coords_row).T] * len(events),
                              ignore_index=True)
        events = pd.concat([df_coords, events], axis=1).round(5)

        return events

    def compute_api(self, precip, days_nb=30, reg=0.8):
        """
        Compute the Antecedent Precipitation Index (API) for a given time series.

        Parameters
        ----------
        precip: np.ndarray
            The precipitation time series.
        days_nb: int
            The number of days to consider for the API calculation.
        reg: float
            The recession constant (between 0 and 1).

        Returns
        -------
        pd.Series
            The computed API values.
        """
        ts_per_day = 24 / self.time_step
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
