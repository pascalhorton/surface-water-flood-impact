"""
Class to compute the impact function with the LSTM + attention model.
"""
from .impact_dl import ImpactDl
from .impact_lstm_options import ImpactLstmOptions
from .impact_lstm_model import ModelLstm
from .impact_cnn_data_generator import ImpactCnnDataGenerator

import copy
import logging
import numpy as np
import pandas as pd

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

DEBUG = False

logger = logging.getLogger(__name__)

# Precipitation is always single-pixel for the LSTM model.
_PRECIP_WINDOW_SIZE = 1
_PRECIP_RESOLUTION = 1


class ImpactLstm(ImpactDl):
    """
    LSTM + multi-head self-attention impact model.

    Precipitation is always loaded at a single pixel (no spatial window).
    The time-series is fed to stacked LSTM layers followed by multi-head
    self-attention before the dense classification head.

    Parameters
    ----------
    options: ImpactLstmOptions
        The model options.
    events: Events
        The events object.
    reload_trained_models: bool
        Whether to reload previously trained models.
    """

    def __init__(self, options, events=None, reload_trained_models=False):
        super().__init__(options, events, reload_trained_models)

        self.dem = None

        if not self.options.is_ok():
            raise ValueError("Options are not ok.")

    def copy(self):
        return copy.deepcopy(self)

    def set_model(self, model):
        """
        Set the model.

        Parameters
        ----------
        model: keras.Model
            The model to set.
        """
        self.model = model

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
            logger.info("Precipitation is not used and is therefore not loaded.")
            return

        precipitation.prepare_data(
            resolution=_PRECIP_RESOLUTION,
            time_step=self.options.precip_time_step
        )

        self.precipitation_hf = precipitation

    def set_dem(self, dem):
        """
        Set the DEM data (single-pixel, no coarsening needed).

        Parameters
        ----------
        dem: xarray.Dataset|None
            The DEM data.
        """
        if dem is None:
            return

        if not self.options.use_precip or not self.options.use_dem:
            logger.info("DEM is not used and is therefore not loaded.")
            return

        assert dem.ndim == 2, "DEM must be 2D"
        self.dem = dem

    def remove_events_without_precipitation_data(self):
        """
        Remove events that fall outside the available precipitation period.
        """
        if self.precipitation_hf is None:
            return

        if 'e_start' in self.df.columns:
            events = self.df[['e_start', 'e_end', 'date_claim']].copy()
            events.rename(columns={'date_claim': 'date'}, inplace=True)
            events['date'] = events['date'].fillna(events[['e_start', 'e_end']].mean(axis=1))
            events['e_start'] = pd.to_datetime(events['e_start']).dt.date
            events['e_end'] = pd.to_datetime(events['e_end']).dt.date
            events['date'] = pd.to_datetime(events['date']).dt.date
        elif 'e_date' in self.df.columns:
            events = self.df[['e_date']].copy()
            events.rename(columns={'e_date': 'date'}, inplace=True)
            events['date'] = pd.to_datetime(events['date']).dt.date
        else:
            raise ValueError("No event date column found in the dataframe.")

        p_start = pd.to_datetime(f'{self.precipitation_hf.year_start}-01-01').date()
        p_end = pd.to_datetime(f'{self.precipitation_hf.year_end}-12-31').date()

        self.df = self.df[events['date'] > p_start + pd.Timedelta(
            days=self.options.precip_days_before)]
        events = events[events['date'] > p_start + pd.Timedelta(
            days=self.options.precip_days_before)]
        self.df = self.df[events['date'] < p_end - pd.Timedelta(
            days=self.options.precip_days_after)]

    def get_data_generator_inference(self, events, features, exposure,
                                     precip_stats=None, mean_static=None,
                                     std_static=None, min_static=None,
                                     max_static=None):
        """
        Build a data generator for inference (no labels).

        Parameters
        ----------
        events: pd.DataFrame
        features: pd.DataFrame|None
        exposure: pd.DataFrame
        precip_stats: xr.Dataset|None
        mean_static, std_static, min_static, max_static: np.array|None

        Returns
        -------
        ImpactCnnDataGenerator
        """
        df = events.merge(exposure, on='cid', how='left')
        if features is not None:
            df = df.merge(features, on='cid', how='left')
            df.dropna(subset=self.features, inplace=True)

        df.dropna(subset=['nb_contracts'], inplace=True)
        df.rename(columns={'i_max_date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        x_static = df[self.features].to_numpy()
        event_props = df[['date', 'x', 'y', 'cid']].to_numpy()

        model_stats = getattr(self, 'model', None)
        if model_stats is not None:
            mean_static = mean_static if mean_static is not None else getattr(model_stats, 'mean_static', None)
            std_static = std_static if std_static is not None else getattr(model_stats, 'std_static', None)
            min_static = min_static if min_static is not None else getattr(model_stats, 'min_static', None)
            max_static = max_static if max_static is not None else getattr(model_stats, 'max_static', None)

        if precip_stats is None:
            mean_precip = getattr(model_stats, 'mean_precip', None) if model_stats is not None else None
            std_precip = getattr(model_stats, 'std_precip', None) if model_stats is not None else None
            q99_precip = getattr(model_stats, 'q99_precip', None) if model_stats is not None else None
        else:
            suffix = '_log' if self.options.log_transform_precip else ''
            mean_precip = self._stats_on_precip_grid(precip_stats, f'mean{suffix}')
            std_precip = self._stats_on_precip_grid(precip_stats, f'std{suffix}')
            q99_precip = self._stats_on_precip_grid(precip_stats, f'q99{suffix}')

        dg = ImpactCnnDataGenerator(
            event_props=event_props,
            x_static=x_static,
            x_precip=self.precipitation_hf,
            x_dem=self.dem,
            batch_size=self.options.batch_size,
            shuffle=False,
            precip_window_size=_PRECIP_WINDOW_SIZE,
            precip_resolution=_PRECIP_RESOLUTION,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=mean_static,
            std_static=std_static,
            min_static=min_static,
            max_static=max_static,
            mean_precip=mean_precip,
            std_precip=std_precip,
            q99_precip=q99_precip,
            debug=DEBUG
        )

        if self.options.use_precip and self.precipitation_hf is not None:
            logger.info("Preloading all precipitation data.")
            all_cids = df['cid'].unique()
            self.precipitation_hf.preload_all_cid_data(all_cids)

        return dg

    def _create_data_generator_train(self):
        self.dg_train = ImpactCnnDataGenerator(
            event_props=self.events_train,
            x_static=self.x_train,
            x_precip=self.precipitation_hf,
            x_dem=self.dem,
            y=self.y_train,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=_PRECIP_WINDOW_SIZE,
            precip_resolution=_PRECIP_RESOLUTION,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            batch_pos_ratio=self.options.batch_pos_ratio,
            debug=DEBUG,
        )

        if self.options.use_precip and self.precipitation_hf is not None:
            logger.info("Preloading all precipitation data.")
            all_cids = self.df['cid'].unique()
            self.precipitation_hf.preload_all_cid_data(all_cids)

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

    def _create_data_generator_valid(self):
        self.dg_val = ImpactCnnDataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation_hf,
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.options.batch_size,
            shuffle=False,
            precip_window_size=_PRECIP_WINDOW_SIZE,
            precip_resolution=_PRECIP_RESOLUTION,
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
            mean_dem=self.dg_train.mean_dem,
            std_dem=self.dg_train.std_dem,
            min_dem=self.dg_train.min_dem,
            max_dem=self.dg_train.max_dem,
            debug=DEBUG
        )

    def _create_data_generator_test(self):
        self.dg_test = ImpactCnnDataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip=self.precipitation_hf,
            x_dem=self.dem,
            y=self.y_test,
            batch_size=self.options.batch_size,
            shuffle=False,
            precip_window_size=_PRECIP_WINDOW_SIZE,
            precip_resolution=_PRECIP_RESOLUTION,
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
            mean_dem=self.dg_train.mean_dem,
            std_dem=self.dg_train.std_dem,
            min_dem=self.dg_train.min_dem,
            max_dem=self.dg_train.max_dem,
            debug=DEBUG
        )

    def _define_model(self):
        """
        Define and build the LSTM+attention Keras model.
        """
        input_1d_size = self.x_train.shape[1:]
        if input_1d_size == (0,):
            input_1d_size = None

        input_3d_size = None
        if self.options.use_precip:
            input_3d_size = [
                self.dg_train.get_time_dim_size(),
                1,
                1,
                self.dg_train.get_nb_channels(),
            ]

        n_pos = np.sum(self.y_train > 0)
        n_neg = np.sum(self.y_train == 0)
        output_bias_init = float(np.log(n_pos / n_neg))

        api_init_idx = -1
        if (self.options.use_api_init and input_1d_size is not None
                and 'api_q' in self.features):
            api_init_idx = list(self.features).index('api_q')
            logger.info("API init: using api_q at feature index %d as LSTM "
                        "initial state.", api_init_idx)
        elif self.options.use_api_init:
            logger.warning("use_api_init=True but api_q is not in features; "
                           "falling back to zero initial state.")

        self.model = ModelLstm(
            task=self.target_type,
            options=self.options,
            input_3d_size=input_3d_size,
            input_1d_size=input_1d_size,
            output_bias_init=output_bias_init,
            api_init_idx=api_init_idx,
        )
        self.model.build_model()

        precip_x, precip_y = self._get_precip_axes()
        self.model.set_feature_stats(
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            q99_precip=self.dg_train.q99_precip,
            precip_x=precip_x,
            precip_y=precip_y,
        )
