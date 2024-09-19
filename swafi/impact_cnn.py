"""
Class to compute the impact function with the CNN model.
"""
from .impact_dl import ImpactDl
from .impact_cnn_options import ImpactCnnOptions
from .impact_cnn_model import ModelCnn
from .utils.data_generator import DataGenerator

import copy

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

DEBUG = False


class ImpactCnn(ImpactDl):
    """
    The CNN Deep Learning Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactCnnOptions
        The model options.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    def __init__(self, events, options, reload_trained_models=False):
        super().__init__(events, options, reload_trained_models)

        self.dem = None

        if not self.options.is_ok():
            raise ValueError("Options are not ok.")

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactCnn
            The copy of the object.
        """
        return copy.deepcopy(self)

    def _create_data_generator_train(self):
        self.dg_train = DataGenerator(
            event_props=self.events_train,
            x_static=self.x_train,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_train,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            debug=DEBUG
        )

        if (self.options.use_precip and self.precipitation is not None and
                self.options.precip_window_size / self.options.precip_resolution == 1):
            print("Preloading all precipitation data.")
            all_cids = self.df['cid'].unique()
            self.precipitation.preload_all_cid_data(all_cids)

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

    def _create_data_generator_valid(self):
        self.dg_val = DataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip=self.dg_train.q99_precip,
            debug=DEBUG
        )

        if self.factor_neg_reduction != 1:
            self.dg_val.reduce_negatives(self.factor_neg_reduction)

    def _create_data_generator_test(self):
        self.dg_test = DataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_test,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip=self.dg_train.q99_precip,
            debug=DEBUG
        )

    def _define_model(self):
        """
        Define the model.
        """
        input_1d_size = self.x_train.shape[1:]
        input_3d_size = None
        pixels_per_side = (self.options.precip_window_size //
                           self.options.precip_resolution)

        if self.options.use_precip:
            input_3d_size = [pixels_per_side,
                             pixels_per_side,
                             self.dg_train.get_third_dim_size(),
                             1] # 1 channel

        self.model = ModelCnn(
            task=self.target_type,
            options=self.options,
            input_3d_size=input_3d_size,
            input_1d_size=input_1d_size,
        )

    def set_precipitation(self, precipitation):
        """
        Set the precipitation data.

        Parameters
        ----------
        precipitation: Precipitation|None
            The precipitation data.
        """
        if precipitation is None:
            return

        if not self.options.use_precip:
            print("Precipitation is not used and is therefore not loaded.")
            return

        precipitation.prepare_data(resolution=self.options.precip_resolution,
                                   time_step=self.options.precip_time_step)

        # Check the shape of the precipitation and the DEM
        if self.dem is not None:
            # Select the same domain as the DEM
            precipitation.generate_pickles_for_subdomain(self.dem.x, self.dem.y)

        self.precipitation = precipitation

    def set_dem(self, dem):
        """
        Set the DEM data.

        Parameters
        ----------
        dem: xarray.Dataset|None
            The DEM data.
        """
        if dem is None:
            return

        if not self.options.use_precip:
            print("DEM is not used and is therefore not loaded.")
            return

        assert dem.ndim == 2, "DEM must be 2D"

        # Adapt the spatial resolution
        if self.options.precip_resolution != 1:
            dem = dem.coarsen(
                x=self.options.precip_resolution,
                y=self.options.precip_resolution,
                boundary='trim'
            ).mean()

        self.dem = dem

    def reduce_spatial_domain(self, precip_window_size):
        """
        Restrict the spatial domain of the precipitation and DEM data.

        Parameters
        ----------
        precip_window_size: int
            The precipitation window size [km].
        """
        precip_window_size_m = 15 * 1000
        if precip_window_size > 15:
            precip_window_size_m = precip_window_size * 1000
        x_min = self.df['x'].min() - precip_window_size_m / 2
        x_max = self.df['x'].max() + precip_window_size_m / 2
        y_min = self.df['y'].min() - precip_window_size_m / 2
        y_max = self.df['y'].max() + precip_window_size_m / 2
        if self.precipitation is not None:
            x_axis = self.precipitation.get_x_axis_for_bounds(x_min, x_max)
            y_axis = self.precipitation.get_y_axis_for_bounds(y_min, y_max)
            self.precipitation.generate_pickles_for_subdomain(x_axis, y_axis)
        if self.dem is not None:
            self.dem = self.dem.sel(
                x=slice(x_min, x_max),
                y=slice(y_max, y_min)
            )
