"""
Class to compute the impact function.
"""

from .config import Config

import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, compute_score_binary


class Impact:
    """
    The generic Impact class.
    """

    def __init__(self, events):
        self.df = events.events
        self.model = None
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
        self.n_jobs = 20
        self.random_state = 42

        # Initialize the data properties
        self._define_potential_features()

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
            elif feature_type == 'land_cover':
                feature_files.append(self.config.get('CSV_FILE_LAND_COVER'))
            elif feature_type == 'runoff_coeff':
                feature_files.append(self.config.get('CSV_FILE_RUNOFF_COEFF'))
            else:
                raise ValueError(f"Unknown file for feature type: {feature_type}")

        # Create unique hash for the data dataframe
        tmp_filename = self._create_data_tmp_file_name(feature_files)

        if tmp_filename.exists():
            print(f"Loading data from {tmp_filename}")
            self.df = pd.read_pickle(tmp_filename)

        else:
            print(f"Creating dataframe and saving to {tmp_filename}")
            for f in feature_files:
                df_features = pd.read_csv(f)

                # Filter out valid column names
                valid_columns = [col for col in self.features
                                 if col in df_features.columns] + ['cid']
                df_features = df_features[valid_columns]

                self.df = self.df.merge(df_features, on='cid', how='left')

            self.df.to_pickle(tmp_filename)

    def split_sample(self, valid_test_size=0.3):
        """
        Split the sample into training, validation and test sets.

        Parameters
        ----------
        valid_test_size: float
            The size of the set for validation and testing (default: 0.3)
        """
        x = self.df[self.features].to_numpy()
        y = self.df['target'].to_numpy().astype(int)

        # Remove lines with NaN values
        x_nan = np.argwhere(np.isnan(x))
        rows_with_nan = np.unique(x_nan[:, 0])
        print(f"Removing {len(rows_with_nan)} rows with NaN values")
        x = np.delete(x, rows_with_nan, axis=0)
        y = np.delete(y, rows_with_nan, axis=0)

        # Split the sample into training and test sets
        self.x_train, x_tmp, self.y_train, y_tmp = train_test_split(
            x, y, test_size=valid_test_size, random_state=self.random_state)
        self.x_test, self.x_valid, self.y_test, self.y_valid = train_test_split(
            x_tmp, y_tmp, test_size=0.5, random_state=self.random_state)

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

    def compute_balanced_class_weights(self):
        """
        Compute balanced the class weights.
        """
        n_classes = len(np.unique(self.y_train))
        self.weights = len(self.y_train) / (n_classes * np.bincount(self.y_train))

    def compute_corrected_class_weights(self, weight_denominator):
        """
        Compute the corrected class weights.
        """
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
                  f"{len(events_with_damages)}")
            print(f"Number of events without damages ({split}): "
                  f"{len(events_without_damages)}")

    def assess_model_on_all_periods(self):
        """
        Assess the model on all periods.
        """
        self._assess_model(self.x_train, self.y_train, 'Train period')
        self._assess_model(self.x_valid, self.y_valid, 'Validation period')
        self._assess_model(self.x_test, self.y_test, 'Test period')

    def _assess_model(self, x, y, period_name):
        """
        Assess the model on a single period.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        y_pred = self.model.predict(x)
        y_pred_prob = self.model.predict_proba(x)

        print(f"\nSplit: {period_name}")

        # Compute the scores
        tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
        print_classic_scores(tp, tn, fp, fn)
        assess_roc_auc(y, y_pred_prob[:, 1])
        print(f"\n----------------------------------------\n")


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
        tag_data = pickle.dumps(feature_files) + pickle.dumps(self.df) + pickle.dumps(
            self.features)
        df_hashed_name = f'data_{hashlib.md5(tag_data).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / df_hashed_name
        return tmp_filename

    def _define_potential_features(self):
        self.tabular_features = {
            'event': ['i_max_q', 'p_sum_q', 'e_tot', 'i_mean_q', 'apireg_q',
                      'nb_contracts'],
            'terrain': ['dem_010m_curv_plan_std', 'dem_010m_slope_min',
                        'dem_010m_curv_plan_mean', 'dem_010m_slope_median',
                        'dem_010m_slope_min'],
            'swf_map': ['area_low', 'area_med', 'area_high'],
            'flowacc': ['dem_025m_flowacc_norivers_max',
                        'dem_010m_flowacc_norivers_max',
                        'dem_050m_flowacc_norivers_max',
                        'dem_100m_flowacc_norivers_max',
                        'dem_010m_flowacc_norivers_median'],
            'land_cover': ['land_cover_cat_7', 'land_cover_cat_11',
                           'land_cover_cat_12'],
            'runoff_coeff': ['runoff_coeff_mean']
        }
