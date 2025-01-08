"""
Class to generate the data for the CNN model.
"""
from .impact_dl_data_generator import ImpactDlDataGenerator

import numpy as np


class ImpactCnnDataGenerator(ImpactDlDataGenerator):
    def __init__(self, event_props, x_static, x_precip, x_dem, y, batch_size=32,
                 shuffle=True, precip_window_size=2, precip_resolution=1,
                 precip_time_step=12, precip_days_before=1, precip_days_after=1,
                 tmp_dir=None, transform_static='standardize', transform_precip='normalize',
                 log_transform_precip=True, mean_static=None, std_static=None,
                 mean_precip=None, std_precip=None, min_static=None,
                 max_static=None, q99_precip=None, debug=False):
        """
        event_props: np.array
            The event properties (2D; dates and coordinates).
        x_static: np.array
            The static predictor variables (0D).
        x_precip: Precipitation
            The precipitation data.
        x_dem: xarray.DataArray
            The DEM (2D).
        y: np.array
            The target variable.
        batch_size: int
            The batch size.
        shuffle: bool
            Whether to shuffle the data or not.
        precip_window_size: int
            The window size for the 3D predictors [km].
        precip_resolution: int
            The desired grid resolution of the precipitation data [km].
        precip_time_step: int
            The desired time step of the precipitation data [h].
        precip_days_before: int
            The number of days before the event to include in the 3D predictors.
        precip_days_after: int
            The number of days after the event to include in the 3D predictors.
        tmp_dir: Path
            The temporary directory to use.
        transform_static: str
            The transformation to apply to the static data.
            Options: 'normalize' or 'standardize'.
        transform_precip: str
            The transformation to apply to the 3D data.
            Options: 'normalize' or 'standardize'.
        log_transform_precip: bool
            Whether to log-transform the precipitation data or not.
        mean_static: np.array
            The mean of the static data.
        std_static: np.array
            The standard deviation of the static data.
        mean_precip: np.array
            The mean of the precipitation data.
        std_precip: np.array
            The standard deviation of the precipitation data.
        min_static: np.array
            The min of the static data.
        max_static: np.array
            The max of the static data.
        q99_precip: np.array
            The 99th percentile of the precipitation data.
        debug: bool
            Whether to run in debug mode or not (print more messages).
        """
        super().__init__(event_props, x_static, y,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         tmp_dir=tmp_dir,
                         transform_static=transform_static,
                         transform_precip=transform_precip,
                         log_transform_precip=log_transform_precip,
                         mean_static=mean_static,
                         std_static=std_static,
                         min_static=min_static,
                         max_static=max_static,
                         debug=debug)
        self.third_dim_size = None
        self.precip_window_size = precip_window_size
        self.precip_resolution = precip_resolution
        self.precip_time_step = precip_time_step
        self.precip_days_before = precip_days_before
        self.precip_days_after = precip_days_after

        self.mean_precip = mean_precip
        self.std_precip = std_precip
        self.q99_precip = q99_precip

        self.mean_dem = None
        self.std_dem = None
        self.min_dem = None
        self.max_dem = None

        self.X_precip = x_precip
        self.X_dem = x_dem

        self._compute_predictor_statistics()

        if transform_static == 'standardize':
            self._standardize_static_inputs()
        elif transform_static == 'normalize':
            self._normalize_static_inputs()

        if transform_precip == 'standardize':
            self._standardize_precip_inputs()
        elif transform_precip == 'normalize':
            self._normalize_precip_inputs()

        self.on_epoch_end()  # Shuffle the data

        if self.X_dem is not None:
            self.X_dem.load()

    def get_third_dim_size(self):
        """ Get the size of the 3rd dimension for the 3D predictors. """
        if self.third_dim_size is not None:
            return self.third_dim_size

        third_dim_size = 0
        if self.X_precip is not None:
            third_dim_size += self.precip_days_after + self.precip_days_before
            third_dim_size *= int(24 / self.precip_time_step)  # Time step
            third_dim_size += 1  # Because the 1st and last time steps are included.
        if self.X_dem is not None:
            third_dim_size += 1  # Add one for the DEM layer

        self.third_dim_size = third_dim_size

        return third_dim_size

    def _standardize_precip_inputs(self):
        if self.X_precip is not None:
            self.X_precip.standardize(self.mean_precip, self.std_precip)
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem

    def _normalize_precip_inputs(self):
        if self.X_precip is not None:
            self.X_precip.normalize(self.q99_precip)
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.min_dem) / (self.max_dem - self.min_dem)

    def _compute_predictor_statistics(self):
        self._compute_static_predictor_statistics()

        if self.X_dem is not None:
            print('Computing DEM predictor statistics')
            if self.transform_precip == 'standardize':
                # Compute the mean and standard deviation of the DEM (non-temporal)
                self.mean_dem = self.X_dem.mean(('x', 'y')).compute().values
                self.std_dem = self.X_dem.std(('x', 'y')).compute().values
            elif self.transform_precip == 'normalize':
                # Compute the min and max of the DEM (non-temporal)
                self.min_dem = self.X_dem.min(('x', 'y')).compute().values
                self.max_dem = self.X_dem.max(('x', 'y')).compute().values

        if self.X_precip is None:
            return

        # Log transform the precipitation
        if self.log_transform_precip:
            print('Log-transforming precipitation')
            self.X_precip.log_transform()

        # Load or compute the precipitation statistics
        if self.transform_precip == 'standardize':
            if self.mean_precip is not None and self.std_precip is not None:
                return
            self.mean_precip, self.std_precip = self.X_precip.compute_mean_and_std_per_pixel()
        elif self.transform_precip == 'normalize':
            if self.q99_precip is not None:
                return
            self.q99_precip = self.X_precip.compute_quantile_per_pixel(0.99)

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        return self._generate_batch(idxs)

    def _generate_batch(self, idxs):
        # Select the events
        y = self.y[idxs]

        x_3d = None
        x_static = None

        # Select the 3D data
        if self.X_precip is not None:
            pixels_nb = int(self.precip_window_size / self.precip_resolution)
            x_3d = np.zeros((len(idxs),
                             pixels_nb,
                             pixels_nb,
                             self.get_third_dim_size()))

            for i_b, event in enumerate(self.event_props[idxs]):
                x_3d[i_b] = self._extract_precipitation(event)

            # Add a new axis for the channels
            x_3d = np.expand_dims(x_3d, axis=-1)

            if self.X_static is None:
                return x_3d, y

        # Select the static data
        if self.X_static is not None:
            x_static = self.X_static[idxs, :]

            if self.X_precip is None:
                return x_static, y

        return (x_3d, x_static), y

    def _extract_precipitation(self, event):
        precip_window_size_m = self.precip_window_size * 1000
        pixels_nb = int(self.precip_window_size / self.precip_resolution)

        # Temporal selection
        t_start = event[0] - np.timedelta64(self.precip_days_before, 'D')
        t_end = event[0] + np.timedelta64(self.precip_days_after, 'D')

        # Spatial domain
        x_start = event[1] - precip_window_size_m / 2
        x_end = event[1] + precip_window_size_m / 2
        y_start = event[2] + precip_window_size_m / 2
        y_end = event[2] - precip_window_size_m / 2

        # Select the corresponding precipitation data
        cid = event[3]
        x_precip_ev = self.X_precip.get_data_chunk(
            t_start, t_end, x_start, x_end, y_start, y_end, cid
        )

        # Move the time axis to the last position
        x_precip_ev = np.moveaxis(x_precip_ev, 0, -1)

        if self.X_dem is not None:
            # Select the corresponding DEM data in the window
            x_dem_ev = self.X_dem.sel(
                x=slice(x_start, x_end),
                y=slice(y_start, y_end)
            ).to_numpy()

            # Replace the NaNs by zeros
            x_dem_ev = np.nan_to_num(x_dem_ev)

            # Add a new axis for the 3rd dimension
            x_dem_ev = np.expand_dims(x_dem_ev, axis=-1)

            # Concatenate
            x_3d_ev = np.concatenate([x_dem_ev, x_precip_ev], axis=-1)
        else:
            x_3d_ev = x_precip_ev

        # If too large, remove the last line(s) or column(s)
        if x_3d_ev.shape[0] > pixels_nb:
            x_3d_ev = x_3d_ev[:pixels_nb, :, :]
        if x_3d_ev.shape[1] > pixels_nb:
            x_3d_ev = x_3d_ev[:, :pixels_nb, :]

        # Handle missing precipitation data
        if x_3d_ev.shape[2] != self.get_third_dim_size():
            self.warning_counter += 1
            self._analyze_precip_shape_difference(
                event, x_3d_ev, x_3d_ev.shape[2], self.get_third_dim_size())

            diff = x_3d_ev.shape[2] - self.get_third_dim_size()
            if abs(diff / self.get_third_dim_size()) > 0.1:  # 10% tolerance
                if self.debug:
                    print(f"Warning: too many missing timesteps ({diff}).")

                pixels_nb = int(self.precip_window_size / self.precip_resolution)
                x_3d_ev = self._create_empty_precip_block(
                    (pixels_nb, pixels_nb, self.get_third_dim_size()))

            else:
                empty_block = self._create_empty_precip_block(
                    (x_3d_ev.shape[0], x_3d_ev.shape[1], -diff))

                x_3d_ev = np.concatenate([x_3d_ev, empty_block], axis=-1)

        return x_3d_ev
