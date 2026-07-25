"""
Class to generate the data for the CNN model.
"""
from .impact_dl_data_generator import ImpactDlDataGenerator
from .precip_archive import get_cdf_levels

import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ImpactCnnDataGenerator(ImpactDlDataGenerator):
    def __init__(self, event_props, x_static, x_precip, x_dem, y=None, batch_size=32,
                 shuffle=True, precip_window_size=2, precip_resolution=1,
                 precip_time_step=60, precip_days_before=1, precip_days_after=1,
                 tmp_dir=None, transform_static='standardize', transform_precip='normalize',
                 log_transform_precip=True, mean_static=None, std_static=None,
                 mean_precip=None, std_precip=None, min_static=None,
                 max_static=None, q99_precip=None, cdf_precip=None,
                 precip_cdf_spread='return_period',
                 mean_dem=None, std_dem=None, min_dem=None, max_dem=None,
                 batch_pos_ratio=None, log_exposure=None, debug=False):
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
            The desired time step of the precipitation data [min]. Must divide
            the day evenly (e.g. 5, 10, 15, 30, 60).
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
        cdf_precip: np.array
            The per-pixel CDF table of the precipitation data (from the training
            generator), used when transform_precip is 'cdf'.
        precip_cdf_spread: str
            How the percentiles are spread over the output range when
            transform_precip is 'cdf' ('return_period' or 'none').
        mean_dem: np.array
            The mean of the DEM data (from training generator).
        std_dem: np.array
            The standard deviation of the DEM data (from training generator).
        min_dem: np.array
            The min of the DEM data (from training generator).
        max_dem: np.array
            The max of the DEM data (from training generator).
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
                         batch_pos_ratio=batch_pos_ratio,
                         log_exposure=log_exposure,
                         debug=debug)
        self.time_dim_size = None
        self.precip_window_size = precip_window_size
        self.precip_resolution = precip_resolution
        self.precip_time_step = precip_time_step
        self.precip_days_before = precip_days_before
        self.precip_days_after = precip_days_after

        self.mean_precip = mean_precip
        self.std_precip = std_precip
        self.q99_precip = q99_precip
        self.cdf_precip = cdf_precip
        self.precip_cdf_spread = precip_cdf_spread
        self.cdf_levels, self.cdf_step = get_cdf_levels(precip_cdf_spread)

        self.mean_dem = mean_dem
        self.std_dem = std_dem
        self.min_dem = min_dem
        self.max_dem = max_dem

        self.X_precip = x_precip
        self.X_dem = x_dem

        self._adapt_event_times()
        self._compute_predictor_statistics()

        if transform_static == 'standardize':
            self._standardize_static_inputs()
        elif transform_static == 'normalize':
            self._normalize_static_inputs()

        if transform_precip == 'standardize':
            self._standardize_precip_inputs()
        elif transform_precip == 'normalize':
            self._normalize_precip_inputs()
        elif transform_precip == 'cdf':
            self._cdf_transform_precip_inputs()

        self.on_epoch_end()  # Shuffle the data

        if self.X_dem is not None:
            self.X_dem.load()

    def get_time_dim_size(self):
        """ Get the number of time steps in the 3D predictors. """
        if self.time_dim_size is not None:
            return self.time_dim_size

        time_dim_size = 0
        if self.X_precip is not None:
            # The time step is in minutes; a step of 1440/minutes must divide the
            # day evenly so the daily grid (and the derived store) are exact.
            minutes = self.precip_time_step
            assert 1440 % minutes == 0, \
                f"The time step ({minutes} min) must divide the day evenly."
            steps_per_day = 1440 // minutes
            time_dim_size += self.precip_days_after + self.precip_days_before + 1
            time_dim_size *= steps_per_day
            time_dim_size += 1  # Because the 1st and last time steps are included.
        self.time_dim_size = time_dim_size

        return time_dim_size

    def get_nb_channels(self):
        """ Get the number of input channels (1 for precip only, 2 when DEM is included). """
        return 2 if self.X_dem is not None else 1

    def _standardize_precip_inputs(self):
        if self.X_precip is not None:
            self.X_precip.standardize(self.mean_precip, self.std_precip)
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem

    def _normalize_precip_inputs(self):
        if self.X_precip is not None:
            self.X_precip.normalize(self.q99_precip)
        self._normalize_dem_input()

    def _cdf_transform_precip_inputs(self):
        if self.X_precip is not None:
            self.X_precip.cdf_transform(
                self.cdf_levels, self.cdf_precip, self.cdf_step)
        # The DEM is not precipitation: it keeps the min-max normalization.
        self._normalize_dem_input()

    def _normalize_dem_input(self):
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.min_dem) / (self.max_dem - self.min_dem)

    def _adapt_event_times(self):
        """
        Adapt the event times to the precipitation time step.
        """
        if self.X_precip is None:
            return

        time_step = f'{self.precip_time_step}min'
        dates = pd.to_datetime(self.event_props[:, 0])
        self.event_props[:, 0] = dates.round(time_step)

    def _compute_predictor_statistics(self):
        self._compute_static_predictor_statistics()

        if self.X_dem is not None:
            if self.transform_precip == 'standardize':
                if self.mean_dem is None or self.std_dem is None:
                    logger.info('Computing DEM predictor statistics')
                    self.mean_dem = self.X_dem.mean(('x', 'y')).compute().values
                    self.std_dem = self.X_dem.std(('x', 'y')).compute().values
            elif self.transform_precip in ['normalize', 'cdf']:
                if self.min_dem is None or self.max_dem is None:
                    logger.info('Computing DEM predictor statistics')
                    self.min_dem = self.X_dem.min(('x', 'y')).compute().values
                    self.max_dem = self.X_dem.max(('x', 'y')).compute().values

        if self.X_precip is None:
            return

        # Log transform the precipitation
        if self.log_transform_precip:
            if self.transform_precip == 'cdf':
                # The CDF transform ranks the values, so any increasing transform
                # applied first leaves its output unchanged.
                logger.info('Skipping the log transform: it has no effect on '
                            'the CDF transform (rank-preserving)')
            else:
                logger.info('Log-transforming precipitation')
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
        elif self.transform_precip == 'cdf':
            if self.cdf_precip is not None:
                return
            self.cdf_precip = self.X_precip.compute_cdf_table_per_pixel(
                self.cdf_levels)

    def __getitem__(self, i):
        """Generate one batch of data"""
        return self._generate_batch(self._get_batch_idxs(i))

    def _generate_batch(self, idxs):
        # Select the events
        y = None
        if self.y is not None:
            y = self.y[idxs]
            # Ensure labels are shaped (batch, 1) for Keras metrics compatibility
            y = np.asarray(y)
            if y.ndim == 1:
                y = np.expand_dims(y, axis=-1)

        x_3d = None
        x_static = None

        # Exposure offset for the Poisson head (last model input when present)
        offset = None
        if self.log_exposure is not None:
            offset = self.log_exposure[idxs].reshape(-1, 1).astype('float32')

        # Select the 3D data
        if self.X_precip is not None:
            pixels_nb = int(self.precip_window_size / self.precip_resolution)
            x_3d = np.zeros((len(idxs),
                             self.get_time_dim_size(),
                             pixels_nb,
                             pixels_nb))

            for i_b, event in enumerate(self.event_props[idxs]):
                x_3d[i_b] = self._extract_precipitation(event)

            if self.X_dem is not None:
                # Build DEM channel: broadcast each static (H, W) patch across T time steps
                x_dem_batch = np.zeros_like(x_3d)
                for i_b, event in enumerate(self.event_props[idxs]):
                    dem_patch = self._extract_dem_patch(event, pixels_nb)  # (H, W)
                    x_dem_batch[i_b] = dem_patch[np.newaxis, :, :]  # broadcast → (T, H, W)
                # Stack precipitation and DEM as separate channels → (batch, T, H, W, 2)
                x_3d = np.stack([x_3d, x_dem_batch], axis=-1)
            else:
                # Single precipitation channel → (batch, T, H, W, 1)
                x_3d = np.expand_dims(x_3d, axis=-1)

            if self.X_static is None or self.X_static.shape[1] == 0:
                if offset is not None:
                    return (x_3d, offset), y
                return x_3d, y

        # Select the static data
        if self.X_static is not None:
            x_static = self.X_static[idxs, :]

            if self.X_precip is None:
                if offset is not None:
                    return (x_static, offset), y
                return x_static, y

        if offset is not None:
            return (x_3d, x_static, offset), y

        return (x_3d, x_static), y

    def _extract_precipitation(self, event):
        """ Extract the precipitation patch for a single event. Returns (T, H, W). """
        precip_window_size_m = self.precip_window_size * 1000
        pixels_nb = int(self.precip_window_size / self.precip_resolution)

        # Temporal selection
        t_start = event[0] - np.timedelta64(self.precip_days_before, 'D')
        t_end = event[0] + np.timedelta64(self.precip_days_after + 1, 'D')  # +1 for the day itself.

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

        # Data arrives as (T, H, W) — keep T first for consistency with model input.

        # If too large, remove the last line(s) or column(s)
        if x_precip_ev.shape[1] > pixels_nb:
            x_precip_ev = x_precip_ev[:, :pixels_nb, :]
        if x_precip_ev.shape[2] > pixels_nb:
            x_precip_ev = x_precip_ev[:, :, :pixels_nb]

        # Handle missing precipitation data
        if x_precip_ev.shape[0] != self.get_time_dim_size():
            self.warning_counter += 1
            self._analyze_precip_shape_difference(
                event, x_precip_ev, x_precip_ev.shape[0], self.get_time_dim_size())

            diff = x_precip_ev.shape[0] - self.get_time_dim_size()
            if abs(diff / self.get_time_dim_size()) > 0.1:  # 10% tolerance
                if self.debug:
                    logger.warning("Too many missing timesteps (%s).", diff)

                x_precip_ev = self._create_empty_precip_block(
                    (self.get_time_dim_size(), pixels_nb, pixels_nb))

            else:
                empty_block = self._create_empty_precip_block(
                    (-diff, x_precip_ev.shape[1], x_precip_ev.shape[2]))

                x_precip_ev = np.concatenate([empty_block, x_precip_ev], axis=0)

        return x_precip_ev

    def _extract_dem_patch(self, event, pixels_nb):
        """ Extract and size-correct the DEM patch for a single event. Returns (H, W). """
        precip_window_size_m = self.precip_window_size * 1000
        x_start = event[1] - precip_window_size_m / 2
        x_end = event[1] + precip_window_size_m / 2
        y_start = event[2] + precip_window_size_m / 2
        y_end = event[2] - precip_window_size_m / 2

        x_dem_ev = self.X_dem.sel(
            x=slice(x_start, x_end),
            y=slice(y_start, y_end)
        ).to_numpy()

        x_dem_ev = np.nan_to_num(x_dem_ev)

        # Trim to expected spatial size if needed
        if x_dem_ev.shape[0] > pixels_nb:
            x_dem_ev = x_dem_ev[:pixels_nb, :]
        if x_dem_ev.shape[1] > pixels_nb:
            x_dem_ev = x_dem_ev[:, :pixels_nb]

        return x_dem_ev
