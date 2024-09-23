"""
Class to compute the impact function with the Transformer model.
"""
from .impact_dl import ImpactDl
from .impact_tx_options import ImpactTransformerOptions
from .impact_tx_model import ModelTransformer
from .impact_tx_data_generator import ImpactTxDataGenerator

import copy
import pandas as pd

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

DEBUG = False


class ImpactTransformer(ImpactDl):
    """
    The Transformer Deep Learning Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactTransformerOptions
        The model options.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    def __init__(self, events, options, reload_trained_models=False):
        super().__init__(events, options, reload_trained_models)

        if not self.options.is_ok():
            raise ValueError("Options are not ok.")

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactTransformer
            The copy of the object.
        """
        return copy.deepcopy(self)

    def _create_data_generator_train(self):
        self.dg_train = ImpactTxDataGenerator(
            event_props=self.events_train,
            x_static=self.x_train,
            x_precip_hf=self.precipitation_hf,
            x_precip_daily=self.precipitation_daily,
            y=self.y_train,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_daily_days_before=self.options.precip_daily_days_before,
            precip_hf_time_step=self.options.precip_hf_time_step,
            precip_hf_days_before=self.options.precip_hf_days_before,
            precip_hf_days_after=self.options.precip_hf_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            debug=DEBUG
        )

        if self.options.use_precip and self.precipitation_hf is not None:
            print("Preloading all high-frequency precipitation data.")
            all_cids = self.df['cid'].unique()
            self.precipitation_hf.preload_all_cid_data(all_cids)

        if self.options.use_precip and self.precipitation_daily is not None:
            print("Preloading all daily precipitation data.")
            all_cids = self.df['cid'].unique()
            self.precipitation_daily.preload_all_cid_data(all_cids)

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

    def _create_data_generator_valid(self):
        self.dg_val = ImpactTxDataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip_hf=self.precipitation_hf,
            x_precip_daily=self.precipitation_daily,
            y=self.y_valid,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_daily_days_before=self.options.precip_daily_days_before,
            precip_hf_time_step=self.options.precip_hf_time_step,
            precip_hf_days_before=self.options.precip_hf_days_before,
            precip_hf_days_after=self.options.precip_hf_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip_hf=self.dg_train.mean_precip_hf,
            std_precip_hf=self.dg_train.std_precip_hf,
            mean_precip_daily=self.dg_train.mean_precip_daily,
            std_precip_daily=self.dg_train.std_precip_daily,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip_hf=self.dg_train.q99_precip_hf,
            q99_precip_daily=self.dg_train.q99_precip_daily,
            debug=DEBUG
        )

        if self.factor_neg_reduction != 1:
            self.dg_val.reduce_negatives(self.factor_neg_reduction)

    def _create_data_generator_test(self):
        self.dg_test = ImpactTxDataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip_hf=self.precipitation_hf,
            x_precip_daily=self.precipitation_daily,
            y=self.y_test,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_daily_days_before=self.options.precip_daily_days_before,
            precip_hf_time_step=self.options.precip_hf_time_step,
            precip_hf_days_before=self.options.precip_hf_days_before,
            precip_hf_days_after=self.options.precip_hf_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip_hf=self.dg_train.mean_precip_hf,
            std_precip_hf=self.dg_train.std_precip_hf,
            mean_precip_daily=self.dg_train.mean_precip_daily,
            std_precip_daily=self.dg_train.std_precip_daily,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip_hf=self.dg_train.q99_precip_hf,
            q99_precip_daily=self.dg_train.q99_precip_daily,
            debug=DEBUG
        )

    def _define_model(self):
        """
        Define the model.
        """
        input_attributes_size = self.x_train.shape[1:]
        input_daily_prec_size = None
        input_high_freq_prec_size = None

        if self.options.use_precip:
            input_high_freq_prec_size = self.dg_train.get_precip_hf_length()
            input_daily_prec_size = self.dg_train.get_precip_daily_length()

        self.model = ModelTransformer(
            task=self.target_type,
            options=self.options,
            input_daily_prec_size=input_daily_prec_size,
            input_high_freq_prec_size=input_high_freq_prec_size,
            input_attributes_size=input_attributes_size,
        )

    def set_precipitation_hf(self, precipitation):
        """
        Set the high-frequency precipitation data.

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

        time_step = self.options.precip_hf_time_step / 60
        precipitation.prepare_data(time_step=time_step)

        self.precipitation_hf = precipitation

    def set_precipitation_daily(self, precipitation):
        """
        Set the daily precipitation data.

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

        precipitation.prepare_data(time_step=24)

        self.precipitation_daily = precipitation

    def reduce_spatial_domain(self):
        """
        Restrict the spatial domain of the precipitation data.
        """
        precip_window_size_m = 1000
        x_min = self.df['x'].min() - precip_window_size_m / 2
        x_max = self.df['x'].max() + precip_window_size_m / 2
        y_min = self.df['y'].min() - precip_window_size_m / 2
        y_max = self.df['y'].max() + precip_window_size_m / 2
        if self.precipitation_hf is not None:
            x_axis = self.precipitation_hf.get_x_axis_for_bounds(x_min, x_max)
            y_axis = self.precipitation_hf.get_y_axis_for_bounds(y_min, y_max)
            self.precipitation_hf.generate_pickles_for_subdomain(x_axis, y_axis)
        if self.precipitation_daily is not None:
            x_axis = self.precipitation_daily.get_x_axis_for_bounds(x_min, x_max)
            y_axis = self.precipitation_daily.get_y_axis_for_bounds(y_min, y_max)
            self.precipitation_daily.generate_pickles_for_subdomain(x_axis, y_axis)

    def remove_events_without_precipitation_data(self):
        """
        Remove the events at the period limits.
        """
        if self.precipitation_hf is None and self.precipitation_daily is None:
            return

        # Extract events dates
        events = self.df[['e_end', 'date_claim']].copy()
        events.rename(columns={'date_claim': 'date'}, inplace=True)
        events['e_end'] = pd.to_datetime(events['e_end']).dt.date
        events['date'] = pd.to_datetime(events['date']).dt.date

        # Fill NaN values with the event end date (as date, not datetime)
        events['date'] = events['date'].fillna(events['e_end'])

        # Precipitation period
        p_hf_start = pd.to_datetime(f'{self.precipitation_hf.year_start}-01-01').date()
        p_hf_end = pd.to_datetime(f'{self.precipitation_hf.year_end}-12-31').date()
        p_daily_start = pd.to_datetime(f'{self.precipitation_daily.year_start}-01-01').date()
        p_daily_end = pd.to_datetime(f'{self.precipitation_daily.year_end}-12-31').date()

        p_start = max(p_hf_start, p_daily_start)
        p_end = min(p_hf_end, p_daily_end)

        precip_days_before = max(self.options.precip_daily_days_before,
                                 self.options.precip_hf_days_before)
        precip_days_after = self.options.precip_hf_days_after

        self.df = self.df[events['date'] > p_start + pd.Timedelta(
            days=precip_days_before)]
        events = events[events['date'] > p_start + pd.Timedelta(
            days=precip_days_before)]
        self.df = self.df[events['date'] < p_end - pd.Timedelta(
            days=precip_days_after)]
