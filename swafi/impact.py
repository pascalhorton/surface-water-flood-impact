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


class Impact:
    """
    The generic Impact class.
    """

    def __init__(self, events):
        self.df = events.events
        self.x_train = None
        self.x_test = None
        self.x_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        self.features = []

        self.config = Config()
        self.tmp_dir = Path(self.config.get('TMP_DIR'))

        # Set default options
        self.weight_denominator = 16

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
        tmp_filename = self._create_tmp_file_name(feature_files)

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

    def split_sample(self, test_size=0.3):
        """
        Split the sample into training, validation and test sets.

        Parameters
        ----------
        test_size: float
            The size of the test set (default: 0.33)
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
            x, y, test_size=test_size, random_state=self.random_state)
        self.x_test, self.x_valid, self.y_test, self.y_valid = train_test_split(
            x_tmp, y_tmp, test_size=0.5, random_state=self.random_state)

    def _create_tmp_file_name(self, feature_files):
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
        tag_data = pickle.dumps(feature_files) + pickle.dumps(self.df) + \
                   pickle.dumps(self.features)
        df_hashed_name = f'data_{hashlib.md5(tag_data).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / df_hashed_name
        return tmp_filename

    def _define_potential_features(self):
        self.tabular_features = {
            'event': ['i_max_q', 'p_sum_q', 'e_tot', 'i_mean_q', 'apireg_q',
                      'i_max', 'p_sum', 'i_mean', 'apireg', 'nb_contracts'],
            'terrain': ['dem_025m_slope_median', 'dem_010m_slope_min',
                        'dem_010m_curv_plan_std', 'dem_025m_curv_plan_std',
                        'dem_010m_curv_plan_mean', 'dem_025m_curv_plan_mean',
                        'dem_050m_curv_plan_mean', 'dem_100m_curv_plan_mean',
                        'dem_100m_slope_median', 'dem_050m_slope_median',
                        'dem_010m_slope_median'],
            'swf_map': ['area_low', 'area_med', 'area_high'],
            'flowacc': ['dem_010m_flowacc_norivers_max',
                        'dem_010m_flowacc_norivers_mean',
                        'dem_010m_flowacc_norivers_median',
                        'dem_025m_flowacc_norivers_max',
                        'dem_025m_flowacc_norivers_mean',
                        'dem_025m_flowacc_norivers_median',
                        'dem_050m_flowacc_norivers_max',
                        'dem_050m_flowacc_norivers_mean',
                        'dem_050m_flowacc_norivers_median',
                        'dem_100m_flowacc_norivers_max',
                        'dem_100m_flowacc_norivers_mean',
                        'dem_100m_flowacc_norivers_median'],
            'land_cover': ['land_cover_cat_1', 'land_cover_cat_2',
                           'land_cover_cat_3', 'land_cover_cat_4',
                           'land_cover_cat_5', 'land_cover_cat_6',
                           'land_cover_cat_7', 'land_cover_cat_8',
                           'land_cover_cat_9', 'land_cover_cat_10',
                           'land_cover_cat_11', 'land_cover_cat_12'],
            'runoff_coeff': ['runoff_coeff_max', 'runoff_coeff_mean']
        }
