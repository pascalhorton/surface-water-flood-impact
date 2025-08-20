"""
Class to compute the impact function.
"""
from .config import Config

import pickle
import hashlib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, store_classic_scores


class Impact:
    """
    The generic Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactBasicOptions|ImpactDlOptions
        The model options.
    """

    def __init__(self, events, options):
        self.options = options
        self.df = events.events
        self.target_type = options.target_type
        self.model = None
        self.events_train = None
        self.events_valid = None
        self.events_test = None
        self.x_train = None
        self.x_test = None
        self.x_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        self.features = []
        self.weights = None
        self.class_weight = None

        self.config = Config()
        self.tmp_dir = Path(self.config.get('TMP_DIR'))

        # Computing options
        self.random_state = options.random_state

        # Initialize the data properties
        self._define_potential_features()

    def select_features(self, features):
        """
        Select the features to use for the model. The features are selected by
        providing a list of feature names with the following format:
        'class:feature'. For example: 'event:i_max_q'.
        This method replaces the default features with the selected ones for the
        selected classes. Classes that are not in the provided list will not be altered.

        Parameters
        ----------
        features: list
            The list of features to use. The features are selected by providing a list
            of feature names with the following format: 'class:feature'. For example:
            'event:i_max_q'.
        """

        # Extract the features/classes mapping from the provided options (class:feature)
        features_selection = {}
        for feature in features:
            if ':' not in feature:
                raise ValueError(f"Invalid feature format: {feature}. "
                                 f"Use 'class:feature'")
            feature_class = feature.split(':')[0]
            feature_name = feature.split(':')[1]
            if feature_class not in self.tabular_features.keys():
                raise ValueError(f"Unknown feature class: {feature_class}")
            if feature_class in features_selection:
                features_selection[feature_class].append(feature_name)
            else:
                features_selection[feature_class] = [feature_name]

        # Replace the features with the selected ones for the selected classes
        for feature_class in features_selection:
            self.tabular_features[feature_class] = features_selection[feature_class]

    def load_features(self, feature_types):
        """
        Load the features from the given feature types.

        Parameters
        ----------
        feature_types: list
            The list of feature types to load. Options are: 'event', 'terrain',
            'swf_map', 'flowacc', 'land_cover', 'runoff_coeff'
        """
        feature_files = []
        for feature_type in feature_types:
            if feature_type not in self.tabular_features.keys():
                raise ValueError(f"Unknown feature type: {feature_type}")

            # Add the features to the list
            self.features += self.tabular_features[feature_type]

            # List files to load
            if feature_type == 'event':
                pass
            elif feature_type == 'terrain':
                feature_files.append(self.config.get('CSV_FILE_TERRAIN'))
            elif feature_type == 'swf_map':
                feature_files.append(self.config.get('CSV_FILE_SWF'))
            elif feature_type == 'flowacc':
                feature_files.append(self.config.get('CSV_FILE_FLOWACC'))
            elif feature_type == 'twi':
                feature_files.append(self.config.get('CSV_FILE_TWI'))
            elif feature_type == 'land_cover':
                feature_files.append(self.config.get('CSV_FILE_LAND_COVER'))
            elif feature_type == 'runoff_coeff':
                feature_files.append(self.config.get('CSV_FILE_RUNOFF_COEFF'))
            else:
                raise ValueError(f"Unknown file for feature type: {feature_type}")

        # Create unique hash for the data dataframe
        tmp_filename = self._create_data_tmp_file_name(feature_files)

        try:
            if tmp_filename.exists():
                print(f"Loading data from {tmp_filename}")
                self.df = pd.read_pickle(tmp_filename)
            else:
                raise FileNotFoundError
        except (pickle.UnpicklingError, FileNotFoundError, EOFError, Exception):
            print(f"Creating dataframe and saving to {tmp_filename}")
            for f in feature_files:
                df_features = pd.read_csv(f)

                # Filter out valid column names
                valid_columns = [col for col in self.features
                                 if col in df_features.columns] + ['cid']
                df_features = df_features[valid_columns]

                self.df = self.df.merge(df_features, on='cid', how='left')

            self.df.to_pickle(tmp_filename)

    def select_nb_contracts_greater_or_equal_to(self, threshold):
        """
        Select only events with a number of contracts above or equal to the given
        threshold.

        Parameters
        ----------
        threshold: int
            The threshold
        """
        self.df = self.df[self.df['nb_contracts'] >= threshold]

    def select_nb_claims_greater_or_equal_to(self, threshold):
        """
        Select only events with a number of claims above or equal to the given
        threshold.

        Parameters
        ----------
        threshold: int
            The threshold
        """
        self.df = self.df[(self.df['nb_claims'] == 0) |
                          (self.df['nb_claims'] >= threshold)]

    def split_sample(self, valid_test_size=0.4, test_size=0.25, ref_date='middle', stratify_by='day'):
        """
        Split the sample into training, validation and test sets. The split is
        stratified on the target, i.e. the proportion of events with and without
        damages will be approximately the same in each split.

        Parameters
        ----------
        valid_test_size: float
            The size of the set for validation and testing (default: 0.4)
        test_size: float
            The size of the set for testing proportionally to the length of the
            validation and testing split (default: 0.25)
        ref_date: str
            The reference date to use for the precipitation extraction when claim dates are missing (no damage class).
            Options are:
            - 'middle' (default): missing dates are filled with the mean of the event start and end date.
            - 'end': missing dates are filled with the event end date.
            - 'i_max': missing dates are filled with the date of the maximum precipitation intensity.
            - 'i_max_only': only the date of the maximum precipitation intensity is used, claim dates are discarded.
        stratify_by: str
            The temporal unit to use for stratification. Options are: 'day' (default) or
            'month'. If 'day', the stratification is done based on days with any damages.
            If 'month', the stratification is done based on monthly damage ratios.
        """
        df = self.df.copy()

        if self.options.min_nb_claims > 1:
            self.df = self.df[(self.df['nb_claims'] == 0) |
                              (self.df['nb_claims'] >= self.options.min_nb_claims)]

        # Set the reference date for the precipitation extraction
        if ref_date == 'middle':
            df.rename(columns={'date_claim': 'date'}, inplace=True)
            # Fill NaN values with the mean of the event start and end date
            df['date'] = df['date'].fillna(df[['e_start', 'e_end']].mean(axis=1))
        elif ref_date == 'end':
            df.rename(columns={'date_claim': 'date'}, inplace=True)
            # Fill NaN values with the event end date
            df['date'] = df['date'].fillna(df['e_end'])
        elif ref_date == 'i_max':
            df.rename(columns={'date_claim': 'date'}, inplace=True)
            # Fill NaN values with the date of the maximum precipitation intensity
            df['date'] = pd.to_datetime(df['i_max_date'])
        elif ref_date == 'i_max_only':
            df.rename(columns={'i_max_date': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError(f"Unknown reference date: {ref_date}. "
                             f"Options are: 'middle', 'i_max'")

        # Transform the dates to a date without time
        df['e_start'] = pd.to_datetime(df['e_start']).dt.date
        df['e_end'] = pd.to_datetime(df['e_end']).dt.date

        # Remove lines with NaN values
        len_before = len(df)
        df.dropna(subset=self.features, inplace=True)
        len_after = len(df)
        print(f"Number of NaN values removed: {len_before - len_after}")

        if stratify_by == 'day':
            # Add a column to flag any claim (1 if there is a damage, 0 otherwise)
            df['damage_class'] = (df['target'] > 0).astype(int)

            # Group all events by date and damage class to split by date without mixing.
            date_label_df = df.groupby('date')['damage_class'].max().reset_index()

            # Split by dates while stratifying on `damage_class`
            train_dates, temp_dates = train_test_split(
                date_label_df['date'],
                test_size=valid_test_size,
                stratify=date_label_df['damage_class'],
                random_state=self.random_state,
                shuffle=True
            )
            val_dates, test_dates = train_test_split(
                temp_dates,
                test_size=test_size,
                stratify=date_label_df.loc[
                    date_label_df['date'].isin(temp_dates), 'damage_class'],
                random_state=self.random_state,
                shuffle=True
            )

            # Filter the original df to get train, validation, and test sets
            train_df = df[df['date'].isin(train_dates)]
            val_df = df[df['date'].isin(val_dates)]
            test_df = df[df['date'].isin(test_dates)]

        elif stratify_by == 'month':
            # Compute the ratio of events with and without damages on an annual basis
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['class'] = np.where(df['target'] > 0, 1, 0)
            events_month = df.groupby(['year', 'month'])['class'].value_counts().unstack(fill_value=0)
            events_month['pos_ratio'] = events_month[1] / (events_month[0] + events_month[1])
            events_month['pos_ratio_ranks'] = events_month['pos_ratio'].rank(method='first')
            events_month['ratio_class'] = pd.cut(events_month['pos_ratio_ranks'], bins=5, labels=False)

            # Split with stratification on the ratio class
            train_slct, tmp_slct = train_test_split(
                events_month,
                test_size=valid_test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=events_month['ratio_class']
            )
            val_slct, test_slct = train_test_split(
                tmp_slct,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=tmp_slct['ratio_class']
            )

            # Filter the original df to get train, validation, and test sets
            train_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(train_slct.index)]
            val_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(val_slct.index)]
            test_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(test_slct.index)]

        else:
            raise ValueError(f"Unknown stratification method: {stratify_by}. "
                             f"Options are: 'day', 'month'")

        self.x_train = train_df[self.features].to_numpy()
        self.x_valid = val_df[self.features].to_numpy()
        self.x_test = test_df[self.features].to_numpy()

        y_fields = ['target', 'date', 'x', 'y', 'cid']
        self.y_train = train_df[y_fields].to_numpy()
        self.y_valid = val_df[y_fields].to_numpy()
        self.y_test = test_df[y_fields].to_numpy()

        # Set the event properties in a separate variable
        self.events_train = self.y_train[:, 1:5]
        self.events_valid = self.y_valid[:, 1:5]
        self.events_test = self.y_test[:, 1:5]
        self.events_train[:, 0] = pd.to_datetime(self.events_train[:, 0])
        self.events_valid[:, 0] = pd.to_datetime(self.events_valid[:, 0])
        self.events_test[:, 0] = pd.to_datetime(self.events_test[:, 0])

        val_y_train = self.y_train[:, 0].astype(float)
        val_y_valid = self.y_valid[:, 0].astype(float)
        val_y_test = self.y_test[:, 0].astype(float)
        if self.target_type == 'occurrence':
            val_y_train = val_y_train.astype(int)
            val_y_valid = val_y_valid.astype(int)
            val_y_test = val_y_test.astype(int)
        self.y_train = val_y_train
        self.y_valid = val_y_valid
        self.y_test = val_y_test

        # Print the percentage of events with and without damages
        self.show_target_stats()
        print(f"Theoretical split ratios: train={100 * (1 - valid_test_size):.1f}%, "
              f"valid={100 * valid_test_size * (1 - test_size):.1f}%, "
              f"test={100 * valid_test_size * test_size:.1f}%")
        y_len = len(self.y_train) + len(self.y_valid) + len(self.y_test)
        print(f"Actual split ratios: train={100 * len(self.y_train) / y_len:.1f}%, "
              f"valid={100 * len(self.y_valid) / y_len:.1f}%, "
              f"test={100 * len(self.y_test) / y_len:.1f}%")

    def normalize_features(self):
        """
        Normalize the features.
        """
        epsilon = 1e-8  # A small constant to avoid division by zero

        # Calculate mean and std only on the training data
        mean = np.mean(self.x_train, axis=0)
        std = np.std(self.x_train, axis=0) + epsilon

        # Normalize all splits using training mean and std
        self.x_train = (self.x_train - mean) / std
        self.x_valid = (self.x_valid - mean) / std
        self.x_test = (self.x_test - mean) / std

    def compute_balanced_class_weights(self, factor_neg_reduction=1):
        """
        Compute balanced the class weights.

        Parameters
        ----------
        factor_neg_reduction: float
            The factor to reduce the number of negative events.
        """
        if self.target_type != 'occurrence':
            raise NotImplemented("Class weights are only available for occurrence")

        n_classes = len(np.unique(self.y_train))
        self.weights = len(self.y_train) / (n_classes * np.bincount(self.y_train))

        # Reduce the number of negative events
        if factor_neg_reduction > 1:
            self.weights[1] /= factor_neg_reduction

    def compute_corrected_class_weights(self, weight_denominator):
        """
        Compute the corrected class weights.
        """
        if self.target_type != 'occurrence':
            raise NotImplemented("Class weights are only available for occurrence")

        self.class_weight = {0: self.weights[0],
                             1: self.weights[1] / weight_denominator}

    def show_target_stats(self):
        # Count the number of events with and without damages
        for split in ['train', 'valid', 'test']:
            y = getattr(self, f'y_{split}')
            if y is None:
                raise ValueError(f"Split {split} not defined")
            events_with_damages = y[y > 0]
            events_without_damages = y[y == 0]
            print(f"Number of events with damages ({split}): "
                  f"({100 * len(events_with_damages) / len(y):.3f}%)"
                  f"({len(events_with_damages)})")
            print(f"Number of events without damages ({split}): "
                  f"({100 * len(events_without_damages) / len(y):.3f}%)"
                  f"({len(events_without_damages)})")

    def create_benchmark_model(self, model_type='random'):
        """
        Create a benchmark model that predicts the occurrence of damages randomly or
        according to other rules.

        Parameters
        ----------
        model_type: str
            The type of benchmark model to create. Options are: 'random', 'always_true',
            'always_false'
        """
        class BenchmarkModel:
            def __init__(self, model_type, target_type):
                self.model_type = model_type
                self.target_type = target_type

            def predict(self, x):
                if self.model_type == 'random':
                    if self.target_type == 'occurrence':
                        return np.array(random.choices([0, 1], k=len(x)))
                    elif self.target_type == 'damage_ratio':
                        return np.random.rand(len(x))
                    else:
                        raise ValueError(f"Unknown target type: {self.target_type}")
                elif self.model_type == 'always_true':
                    return np.ones(len(x))
                elif self.model_type == 'always_false':
                    return np.zeros(len(x))
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

            def predict_proba(self, x):
                if self.model_type == 'random':
                    return np.random.rand(len(x), 2)
                elif self.model_type == 'always_true':
                    return np.ones((len(x), 2))
                elif self.model_type == 'always_false':
                    return np.zeros((len(x), 2))
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

        self.model = BenchmarkModel(model_type, self.target_type)

    def assess_model_on_all_periods(self, save_results=False, file_tag=''):
        """
        Assess the model on all periods.

        Parameters
        ----------
        save_results: bool
            Save the results to a file.
        file_tag: str
            The tag to add to the file name.
        """
        df_res = pd.DataFrame(columns=['split'])
        df_res = self._assess_model(self.x_train, self.y_train, 'train', df_res)
        df_res = self._assess_model(self.x_valid, self.y_valid, 'valid', df_res)
        df_res = self._assess_model(self.x_test, self.y_test, 'test', df_res)

        if save_results:
            self._save_results_csv(df_res, file_tag)

    def _save_results_csv(self, df_res, file_tag):
        output_dir = self.config.output_dir
        date_tag = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M%S')
        dataset = self.options.dataset
        seed_tag = ''
        if self.random_state is not None:
            seed_tag = f'_seed_{self.random_state}'
        base_name = f'results_{dataset}_{file_tag}{seed_tag}_{date_tag}'
        file_name = f'{output_dir}/{base_name}.csv'
        df_res.to_csv(file_name, index=False)
        file_name_options = f'{output_dir}/{base_name}_options.csv'
        df_options = pd.DataFrame(self.options.__dict__.items(), columns=['option', 'value'])
        df_options.to_csv(file_name_options, index=False)
        print(f"Results saved to {file_name}")

    def _assess_model(self, x, y, period_name, df_res):
        """
        Assess the model on a single period.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        y_pred = self.model.predict(x)

        print(f"\nSplit: {period_name}")

        df_tmp = pd.DataFrame(columns=df_res.columns)
        df_tmp['split'] = [period_name]

        # Compute the scores
        if self.target_type == 'occurrence':
            tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
            print_classic_scores(tp, tn, fp, fn)
            store_classic_scores(tp, tn, fp, fn, df_tmp)
            y_pred_prob = self.model.predict_proba(x)
            roc = assess_roc_auc(y, y_pred_prob[:, 1])
            df_tmp['ROC_AUC'] = [roc]
        else:
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            print(f"RMSE: {rmse}")
            df_tmp['RMSE'] = [rmse]
        print(f"----------------------------------------")

        df_res = pd.concat([df_res, df_tmp])

        return df_res

    def _create_data_tmp_file_name(self, feature_files):
        """
        Create a unique file name for the given features.

        Parameters
        ----------
        feature_files: list
            The list of feature files

        Returns
        -------
        Path
            The unique file name
        """
        # Create unique hash for the data dataframe
        tag_data = (pickle.dumps(feature_files) + pickle.dumps(self.df.shape) +
                    pickle.dumps(self.df.columns) + pickle.dumps(self.df.iloc[0]) +
                    pickle.dumps(self.features))
        df_hashed_name = f'data_{hashlib.md5(tag_data).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / df_hashed_name
        return tmp_filename

    def _define_potential_features(self):
        self.tabular_features = {}

        if self.options.use_event_attributes:
            self.tabular_features['event'] = [
                'i_max_q', 'p_sum_q', 'duration', 'i_mean_q',
                'api_q', 'nb_contracts']

        if self.options.use_static_attributes:
            if not self.options.use_all_static_attributes:
                self.tabular_features['terrain'] = [
                    'dem_050m_curv_plan_median', 'dem_050m_slope_min',
                    'dem_100m_curv_plan_median', 'dem_100m_slope_min',
                    'dem_250m_curv_plan_median', 'dem_250m_slope_min']
                self.tabular_features['swf_map'] = [
                    'area_low', 'area_med', 'area_high',
                    'n_buildings_low', 'n_buildings_med', 'n_buildings_high']
                self.tabular_features['flowacc'] = [
                    'dem_100m_flowacc_median', 'dem_250m_flowacc_median']
                self.tabular_features['twi'] = [
                    'dem_010m_twi_max', 'dem_010m_twi_median']
                self.tabular_features['land_cover'] = [
                    'land_cover_cat_7', 'land_cover_cat_11', 'land_cover_cat_12']
            else:
                self.tabular_features['terrain'] = [
                    'dem_010m_curv_plan_min', 'dem_010m_curv_plan_max',
                    'dem_010m_curv_plan_mean', 'dem_010m_curv_plan_std',
                    'dem_010m_curv_plan_median', 'dem_010m_curv_prof_min',
                    'dem_010m_curv_prof_max', 'dem_010m_curv_prof_mean',
                    'dem_010m_curv_prof_std', 'dem_010m_curv_prof_median',
                    'dem_010m_curv_tot_min', 'dem_010m_curv_tot_max',
                    'dem_010m_curv_tot_mean', 'dem_010m_curv_tot_std',
                    'dem_010m_curv_tot_median', 'dem_010m_slope_min',
                    'dem_010m_slope_max', 'dem_010m_slope_mean',
                    'dem_010m_slope_std', 'dem_010m_slope_median',
                    'dem_025m_curv_plan_min', 'dem_025m_curv_plan_max',
                    'dem_025m_curv_plan_mean', 'dem_025m_curv_plan_std',
                    'dem_025m_curv_plan_median', 'dem_025m_curv_prof_min',
                    'dem_025m_curv_prof_max', 'dem_025m_curv_prof_mean',
                    'dem_025m_curv_prof_std', 'dem_025m_curv_prof_median',
                    'dem_025m_curv_tot_min', 'dem_025m_curv_tot_max',
                    'dem_025m_curv_tot_mean', 'dem_025m_curv_tot_std',
                    'dem_025m_curv_tot_median', 'dem_025m_slope_min',
                    'dem_025m_slope_max', 'dem_025m_slope_mean',
                    'dem_025m_slope_std', 'dem_025m_slope_median',
                    'dem_050m_curv_plan_min', 'dem_050m_curv_plan_max',
                    'dem_050m_curv_plan_mean', 'dem_050m_curv_plan_std',
                    'dem_050m_curv_plan_median', 'dem_050m_curv_prof_min',
                    'dem_050m_curv_prof_max', 'dem_050m_curv_prof_mean',
                    'dem_050m_curv_prof_std', 'dem_050m_curv_prof_median',
                    'dem_050m_curv_tot_min', 'dem_050m_curv_tot_max',
                    'dem_050m_curv_tot_mean', 'dem_050m_curv_tot_std',
                    'dem_050m_curv_tot_median', 'dem_050m_slope_min',
                    'dem_050m_slope_max', 'dem_050m_slope_mean',
                    'dem_050m_slope_std', 'dem_050m_slope_median',
                    'dem_100m_curv_plan_min', 'dem_100m_curv_plan_max',
                    'dem_100m_curv_plan_mean', 'dem_100m_curv_plan_std',
                    'dem_100m_curv_plan_median', 'dem_100m_curv_prof_min',
                    'dem_100m_curv_prof_max', 'dem_100m_curv_prof_mean',
                    'dem_100m_curv_prof_std', 'dem_100m_curv_prof_median',
                    'dem_100m_curv_tot_min', 'dem_100m_curv_tot_max',
                    'dem_100m_curv_tot_mean', 'dem_100m_curv_tot_std',
                    'dem_100m_curv_tot_median', 'dem_100m_slope_min',
                    'dem_100m_slope_max', 'dem_100m_slope_mean',
                    'dem_100m_slope_std', 'dem_100m_slope_median',
                    'dem_250m_curv_plan_min', 'dem_250m_curv_plan_max',
                    'dem_250m_curv_plan_mean', 'dem_250m_curv_plan_std',
                    'dem_250m_curv_plan_median', 'dem_250m_curv_prof_min',
                    'dem_250m_curv_prof_max', 'dem_250m_curv_prof_mean',
                    'dem_250m_curv_prof_std', 'dem_250m_curv_prof_median',
                    'dem_250m_curv_tot_min', 'dem_250m_curv_tot_max',
                    'dem_250m_curv_tot_mean', 'dem_250m_curv_tot_std',
                    'dem_250m_curv_tot_median', 'dem_250m_slope_min',
                    'dem_250m_slope_max', 'dem_250m_slope_mean',
                    'dem_250m_slope_std', 'dem_250m_slope_median']
                self.tabular_features['swf_map'] = [
                    'area_low', 'area_med', 'area_high', 'area_exposed',
                    'n_buildings_low', 'n_buildings_med', 'n_buildings_high',
                    'n_buildings_exposed']
                self.tabular_features['flowacc'] = [
                    'dem_010m_flowacc_max', 'dem_010m_flowacc_mean',
                    'dem_010m_flowacc_std', 'dem_010m_flowacc_median',
                    'dem_025m_flowacc_max', 'dem_025m_flowacc_mean',
                    'dem_025m_flowacc_std', 'dem_025m_flowacc_median',
                    'dem_050m_flowacc_max', 'dem_050m_flowacc_mean',
                    'dem_050m_flowacc_std', 'dem_050m_flowacc_median',
                    'dem_100m_flowacc_max', 'dem_100m_flowacc_mean',
                    'dem_100m_flowacc_std', 'dem_100m_flowacc_median',
                    'dem_250m_flowacc_max', 'dem_250m_flowacc_mean',
                    'dem_250m_flowacc_std', 'dem_250m_flowacc_median']
                self.tabular_features['twi'] = [
                    'dem_010m_twi_max', 'dem_010m_twi_mean',
                    'dem_010m_twi_std', 'dem_010m_twi_median']
                self.tabular_features['land_cover'] = [
                    'land_cover_cat_7', 'land_cover_cat_11',
                    'land_cover_cat_12']

