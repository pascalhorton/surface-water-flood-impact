"""
Class to compute the impact function.
"""
import logging

from .config import Config

import pickle
import hashlib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve

from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_pr_auc, assess_recall_at_budget, assess_roc_auc, store_classic_scores

logger = logging.getLogger(__name__)


class Impact:
    """
    The generic Impact class.

    Parameters
    ----------
    options: ImpactBasicOptions|ImpactDlOptions
        The model options.
    events: Events
        The events object.
    """

    def __init__(self, options, events=None):
        self.options = options
        self.df = events.events if events is not None else None
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
        self.exposure_train = None
        self.exposure_valid = None
        self.exposure_test = None
        self.features = []
        self.weights = None
        self.class_weight = None
        self.probability_threshold = 0.5
        self.x_mean = None
        self.x_std = None

        self.config = Config()
        self.tmp_dir = Path(self.config.get('TMP_DIR'))

        # Computing options
        self.random_state = options.random_state

        # Initialize the data properties
        events_columns = events.events.columns if events is not None else []
        self._define_potential_features(events_columns)

    def update_potential_features(self, events_columns):
        """
        Re-derive the default tabular features from the given events columns.

        Needed when the model was constructed without events (e.g. in the
        inference scripts): the sub-hourly event features (p_10min_q, ...) are
        only part of the defaults when present in the events, so the defaults
        must be re-aligned with the events actually used before selecting
        features.

        Parameters
        ----------
        events_columns: list|pd.Index
            The column names of the events dataframe.
        """
        self._define_potential_features(events_columns)

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

    def get_feature_files(self, feature_types):
        """
        Get the list of feature files to load based on the selected feature types.

        Parameters
        ----------
        feature_types: list
            The list of feature types to load. Options are: 'event', 'terrain',
            'swf_map', 'flowacc', 'land_cover', 'runoff_coeff'

        Returns
        -------
        list
            The list of feature files to load.
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

        return feature_files

    def get_all_features(self, feature_types):
        """
        Get all features from the given feature files.

        Parameters
        ----------
        feature_types: list
            The list of feature types to load. Options are: 'event', 'terrain',
            'swf_map', 'flowacc', 'land_cover', 'runoff_coeff'

        Returns
        -------
        pd.DataFrame
            The dataframe with all features.
        """
        feature_files = self.get_feature_files(feature_types)

        all_features = None

        for f in feature_files:
            df_features = pd.read_csv(f)

            # Filter out valid column names
            valid_columns = [col for col in self.features
                             if col in df_features.columns] + ['cid']
            df_features = df_features[valid_columns]

            if all_features is None:
                all_features = df_features
            else:
                all_features = all_features.merge(df_features, on='cid', how='left')

        return all_features

    def load_features(self, feature_types, use_pickle=False):
        """
        Load the features from the given feature types.

        Parameters
        ----------
        feature_types: list
            The list of feature types to load. Options are: 'event', 'terrain',
            'swf_map', 'flowacc', 'land_cover', 'runoff_coeff'
        use_pickle: bool
            Whether to load the features from a pickle file if it exists. If False,
            the features will be loaded from the CSV files.
        """
        feature_files = self.get_feature_files(feature_types)

        # Create unique hash for the data dataframe
        tmp_filename = self._create_data_tmp_file_name(feature_files)

        try:
            if use_pickle and tmp_filename.exists():
                logger.info("Loading data from %s", tmp_filename)
                self.df = pd.read_pickle(tmp_filename)
            else:
                raise FileNotFoundError
        except (pickle.UnpicklingError, FileNotFoundError, EOFError, Exception):
            for f in feature_files:
                df_features = pd.read_csv(f)

                # Filter out valid column names
                valid_columns = [col for col in self.features
                                 if col in df_features.columns] + ['cid']
                df_features = df_features[valid_columns]

                self.df = self.df.merge(df_features, on='cid', how='left')

            if use_pickle:
                logger.info("Saving dataframe to %s", tmp_filename)
                self.df.to_pickle(tmp_filename)

    def set_events(self, events):
        """
        Set the events dataframe.

        Parameters
        ----------
        events: pd.DataFrame
            The events dataframe.
        """
        self.df = events

    def set_features(self, features):
        """
        Set the features to use for the model.

        Parameters
        ----------
        features: pd.DataFrame
            The features dataframe. Must contain a 'cid' column to merge with the
            events dataframe.
        """
        self.df = self.df.merge(features, on='cid', how='left')

    def set_exposure(self, exposure):
        """
        Set the exposure dataframe.

        Parameters
        ----------
        exposure: pd.DataFrame
            The exposure dataframe. Must contain a 'cid' column to merge with the
            events dataframe.
        """
        self.df = self.df.merge(exposure, on='cid', how='left')

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

    def split_sample(self, valid_test_size=0.3, test_size=0, ref_date='i_max_only',
                     split_mode='chronological'):
        """
        Split the sample into training, validation and test sets.

        Parameters
        ----------
        valid_test_size: float
            The size of the set for validation and testing (default: 0.25)
        test_size: float
            The size of the set for testing proportionally to the length of the
            validation and testing split (default: 0)
        ref_date: str
            The reference date to use for the precipitation extraction when claim dates are missing (no damage class).
            Options are:
            - 'middle': missing dates are filled with the mean of the event start and end date.
            - 'end': missing dates are filled with the event end date.
            - 'i_max': missing dates are filled with the date of the maximum precipitation intensity.
            - 'i_max_only' (default): only the date of the maximum precipitation intensity is used, claim dates are discarded.
        split_mode: str
            How to assign the days to the splits. Options are:
            - 'chronological' (default): the last days of the period are held
              out. Validation then measures what the model is actually asked to
              do — generalise to a later period — so both the scores and the
              probability threshold tuned on it transfer to unseen years.
            - 'random_days': random days, stratified on whether the day carries
              a claim. Days from the whole period are interleaved between the
              splits, so validation shares the climate of the training set and
              cannot see any drift between periods.
            - 'random_months': random (year, month) blocks, stratified on the
              monthly damage ratio.
            A whole day always lands in a single split, in every mode: the
            events of one day share a storm across many cells.
        """
        df = self.df.copy()

        if self.options.min_nb_claims > 1:
            self.df = self.df[(self.df['nb_claims'] == 0) |
                              (self.df['nb_claims'] >= self.options.min_nb_claims)]

        if 'e_date' in df.columns:
            # Simple event definition
            df.rename(columns={'e_date': 'date'}, inplace=True)
        else:

            # Set the reference date for the precipitation extraction. We force the time to 18:00 to
            # avoid overfitting on the time of the day.
            if ref_date == 'middle':
                df.rename(columns={'date_claim': 'date'}, inplace=True)
                # Set a time to the claim date (18:00 by default)
                df['date'] = pd.to_datetime(df['date'], errors='coerce') + pd.Timedelta(hours=18)
                # Fill NaN values with the mean of the event start and end date
                fill_datetime = (pd.to_datetime(df['e_start']) + pd.to_datetime(df['e_end'])) / 2
                fill_datetime = fill_datetime.dt.floor('D') + pd.Timedelta(hours=18)
                df['date'] = df['date'].fillna(fill_datetime)
            elif ref_date == 'end':
                df.rename(columns={'date_claim': 'date'}, inplace=True)
                # Set a time to the claim date (18:00 by default)
                df['date'] = pd.to_datetime(df['date'], errors='coerce') + pd.Timedelta(hours=18)
                # Fill NaN values with the event end date
                fill_datetime = pd.to_datetime(df['e_end']).dt.floor('D') + pd.Timedelta(hours=18)
                df['date'] = df['date'].fillna(fill_datetime)
            elif ref_date == 'i_max':
                df.rename(columns={'date_claim': 'date'}, inplace=True)
                # Set a time to the claim date (18:00 by default)
                df['date'] = pd.to_datetime(df['date'], errors='coerce') + pd.Timedelta(hours=18)
                # Fill NaN values with the date of the maximum precipitation intensity
                fill_datetime = pd.to_datetime(df['i_max_date']).dt.floor('D') + pd.Timedelta(hours=18)
                df['date'] = df['date'].fillna(fill_datetime)
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
        logger.info("Number of NaN values removed: %s", len_before - len_after)

        if split_mode == 'chronological':
            # Hold out the last days of the period. Splitting on the calendar
            # day (not on the rows) keeps all the events of a day together.
            days = np.sort(df['date'].dt.floor('D').unique())
            n_held_out = int(round(len(days) * valid_test_size))
            assert n_held_out > 0, "The validation split is empty."
            train_days = days[:len(days) - n_held_out]
            held_out = days[len(days) - n_held_out:]
            if test_size == 0:
                valid_days, test_days = held_out, held_out[:0]
            else:
                n_test = int(round(len(held_out) * test_size))
                valid_days, test_days = held_out[:len(held_out) - n_test], \
                    held_out[len(held_out) - n_test:]

            day = df['date'].dt.floor('D')
            train_df = df[day.isin(train_days)]
            val_df = df[day.isin(valid_days)]
            test_df = df[day.isin(test_days)]

            for name, days_split in (('train', train_days), ('valid', valid_days),
                                     ('test', test_days)):
                if len(days_split) > 0:
                    logger.info("Split %s: %s to %s (%d days)", name,
                                pd.Timestamp(days_split[0]).date(),
                                pd.Timestamp(days_split[-1]).date(), len(days_split))

        elif split_mode == 'random_days':
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
            if test_size == 0:
                val_df = df[df['date'].isin(temp_dates)]
                test_df = df.iloc[0:0]
            else:
                val_dates, test_dates = train_test_split(
                    temp_dates,
                    test_size=test_size,
                    stratify=date_label_df.loc[
                        date_label_df['date'].isin(temp_dates), 'damage_class'],
                    random_state=self.random_state,
                    shuffle=True
                )
                val_df = df[df['date'].isin(val_dates)]
                test_df = df[df['date'].isin(test_dates)]

            # Filter training set
            train_df = df[df['date'].isin(train_dates)]

        elif split_mode == 'random_months':
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
            # Filter the original df to get train, validation, and test sets
            train_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(train_slct.index)]
            if test_size == 0:
                val_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(tmp_slct.index)]
                test_df = df.iloc[0:0]
            else:
                val_slct, test_slct = train_test_split(
                    tmp_slct,
                    test_size=test_size,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=tmp_slct['ratio_class']
                )
                val_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(val_slct.index)]
                test_df = df[pd.MultiIndex.from_arrays([df['year'], df['month']]).isin(test_slct.index)]

        else:
            raise ValueError(f"Unknown split mode: {split_mode}. Options are: "
                             f"'chronological', 'random_days', 'random_months'")

        self.x_train = train_df[self.features].to_numpy()
        self.x_valid = val_df[self.features].to_numpy()
        self.x_test = test_df[self.features].to_numpy()

        y_fields = ['target', 'date', 'x', 'y', 'cid']
        use_poisson_head = getattr(self.options, 'use_poisson_head', False)
        if use_poisson_head:
            y_fields += ['nb_claims', 'nb_contracts']
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

        if use_poisson_head:
            # The response is the claim count; log(nb_contracts) is the exposure
            # offset fed to the model as an extra input.
            val_y_train = self.y_train[:, 5].astype(float).astype(int)
            val_y_valid = self.y_valid[:, 5].astype(float).astype(int)
            val_y_test = self.y_test[:, 5].astype(float).astype(int)
            self.exposure_train = np.log(self.y_train[:, 6].astype(float))
            self.exposure_valid = np.log(self.y_valid[:, 6].astype(float))
            self.exposure_test = np.log(self.y_test[:, 6].astype(float))
        else:
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
        if test_size == 0:
            logger.info("Theoretical split ratios: train=%.1f%%, valid=%.1f%%",
                        100 * (1 - valid_test_size),
                        100 * valid_test_size)
            y_len = len(self.y_train) + len(self.y_valid)
            logger.info("Actual split ratios: train=%.1f%%, valid=%.1f%%",
                        100 * len(self.y_train) / y_len,
                        100 * len(self.y_valid) / y_len)
        else:
            logger.info("Theoretical split ratios: train=%.1f%%, valid=%.1f%%, test=%.1f%%",
                        100 * (1 - valid_test_size),
                        100 * valid_test_size * (1 - test_size),
                        100 * valid_test_size * test_size)
            y_len = len(self.y_train) + len(self.y_valid) + len(self.y_test)
            logger.info("Actual split ratios: train=%.1f%%, valid=%.1f%%, test=%.1f%%",
                        100 * len(self.y_train) / y_len,
                        100 * len(self.y_valid) / y_len,
                        100 * len(self.y_test) / y_len)

    def merge_valid_test_into_train(self):
        """
        Merge the validation (and test) splits back into the training set, so a
        subsequent fit() uses the whole period. The validation and test splits
        are emptied.

        Intended for a final, deployment model refit on all the available data
        once the hyperparameters and the decision threshold have been selected
        on the held-out split: assessment and threshold tuning must therefore
        already be done, as no held-out data remains afterwards.
        """
        self.x_train = np.concatenate(
            [self.x_train, self.x_valid, self.x_test], axis=0)
        self.y_train = np.concatenate(
            [self.y_train, self.y_valid, self.y_test], axis=0)
        self.events_train = np.concatenate(
            [self.events_train, self.events_valid, self.events_test], axis=0)

        # Empty the held-out splits, keeping their shape and dtype.
        self.x_valid, self.x_test = self.x_train[:0], self.x_train[:0]
        self.y_valid, self.y_test = self.y_train[:0], self.y_train[:0]
        self.events_valid = self.events_train[:0]
        self.events_test = self.events_train[:0]

        logger.info("Merged all splits into the training set: %d samples.",
                    len(self.y_train))

    def normalize_features(self):
        """
        Normalize the features.
        """
        epsilon = 1e-8  # A small constant to avoid division by zero

        # Calculate mean and std only on the training data
        self.x_mean = np.mean(self.x_train, axis=0)
        self.x_std = np.std(self.x_train, axis=0) + epsilon

        # Normalize all splits using training mean and std
        self.x_train = (self.x_train - self.x_mean) / self.x_std
        self.x_valid = (self.x_valid - self.x_mean) / self.x_std
        self.x_test = (self.x_test - self.x_mean) / self.x_std

    def compute_average_precision(self, x, y):
        """
        Compute the average precision (area under the precision-recall curve) on
        the given set. This is a threshold-free metric well suited to imbalanced
        occurrence problems, making it a more stable objective for hyperparameter
        optimization than the F1 score at a fixed threshold.

        Parameters
        ----------
        x: np.array
            The features.
        y: np.array
            The target.

        Returns
        -------
        float
            The average precision score.
        """
        if self.target_type != 'occurrence':
            raise NotImplementedError(
                "Average precision is only available for occurrence")

        y_prob = self.model.predict_proba(x)[:, 1]
        return average_precision_score(y, y_prob)

    def tune_probability_threshold(self, x=None, y=None):
        """
        Find the decision threshold that maximizes the F1 score on the given set
        (the validation set by default) and store it in
        ``self.probability_threshold``. The threshold is subsequently used when
        turning predicted probabilities into class labels (assessment, inference).

        Parameters
        ----------
        x: np.array
            The features. If None, the validation set is used.
        y: np.array
            The target. If None, the validation set is used.

        Returns
        -------
        float
            The selected probability threshold.
        """
        if self.target_type != 'occurrence':
            raise NotImplementedError(
                "Threshold tuning is only available for occurrence")

        if x is None or y is None:
            x, y = self.x_valid, self.y_valid

        y_prob = self.model.predict_proba(x)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y, y_prob)
        # precision/recall have one more element than thresholds (the last point
        # corresponds to recall=0 with no threshold); drop it before scoring.
        epsilon = 1e-7
        f1 = 2 * precision[:-1] * recall[:-1] / (
                precision[:-1] + recall[:-1] + epsilon)

        best_idx = int(np.argmax(f1))
        self.probability_threshold = float(thresholds[best_idx])

        logger.info(
            "Tuned probability threshold: %.4f (valid F1=%.4f)",
            self.probability_threshold, f1[best_idx])

        return self.probability_threshold

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

        When batch_pos_ratio is set, the denominator is automatically scaled so
        that the gradient contributions of positives and negatives are balanced at
        the batch level. `weight_denominator` then acts as a fine-tuning multiplier
        around that balanced point (1 = perfectly balanced, >1 = favour negatives).
        """
        if self.target_type != 'occurrence':
            raise NotImplemented("Class weights are only available for occurrence")

        batch_pos_ratio = getattr(self.options, 'batch_pos_ratio', None)

        if batch_pos_ratio is not None:
            # d_balanced = (p_neg/p_pos) * batch_pos_ratio / (1 - batch_pos_ratio)
            # gives gradient ratio pos:neg == 1:1 for the given batch composition.
            n_pos = np.sum(self.y_train > 0)
            n_neg = np.sum(self.y_train == 0)
            p_pos = n_pos / len(self.y_train)
            p_neg = n_neg / len(self.y_train)
            d_balanced = (p_neg / p_pos) * (batch_pos_ratio / (1.0 - batch_pos_ratio))
            effective_denom = d_balanced * weight_denominator
            logger.info(
                "batch_pos_ratio=%.3f: balanced denominator=%.1f, "
                "weight_denominator=%.1f -> effective denominator=%.1f",
                batch_pos_ratio, d_balanced, weight_denominator, effective_denom)
        else:
            effective_denom = weight_denominator

        self.class_weight = {0: self.weights[0],
                             1: self.weights[1] / effective_denom}
        ratio = self.class_weight[1] / self.class_weight[0]
        logger.info(
            "Class weights: neg=%.4f, pos=%.4f (ratio pos/neg=%.2f)",
            self.class_weight[0], self.class_weight[1], ratio)

        # compute_balanced_class_weights() already divided the positive weight by
        # factor_neg_reduction, so weights[1] is the value that balances the
        # subsampled training generator. effective_denom is therefore exactly the
        # factor by which the negatives are made to outweigh the positives.
        if effective_denom > 1:
            logger.warning(
                "Negatives carry about %.0f× the loss mass of the positives "
                "(effective denominator %.1f) on a problem with %d positives and "
                "%d negatives. The constant 'no event' prediction sits close to "
                "the loss minimum, and the model may collapse onto it. Set "
                "--weight-denominator 1 for balanced weighting (currently %s).",
                effective_denom, effective_denom,
                int(np.sum(self.y_train > 0)), int(np.sum(self.y_train == 0)),
                weight_denominator)

    def show_target_stats(self):
        # Count the number of events with and without damages
        for split in ['train', 'valid', 'test']:
            y = getattr(self, f'y_{split}')
            if y is None:
                raise ValueError(f"Split {split} not defined")
            if split == 'test' and len(y) == 0:
                continue
            events_with_damages = y[y > 0]
            events_without_damages = y[y == 0]
            logger.info("Number of events with damages (%s): (%.3f%%)(%s)",
                        split, 100 * len(events_with_damages) / len(y), len(events_with_damages))
            logger.info("Number of events without damages (%s): (%.3f%%)(%s)",
                        split, 100 * len(events_without_damages) / len(y), len(events_without_damages))

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
        logger.info("Assessing the model on all periods.")
        df_res = pd.DataFrame(columns=['split'])
        df_res = self._assess_model(self.x_train, self.y_train, 'train', df_res)
        df_res = self._assess_model(self.x_valid, self.y_valid, 'valid', df_res)
        if self.y_test is not None and len(self.y_test) > 0:
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
        logger.info("Results saved to %s", file_name)

    @staticmethod
    def _flag_degenerate_predictions(y_pred, roc, tp, tn, fp, fn, period_name):
        """
        Mark a row whose threshold-derived scores describe the threshold search
        rather than the model, so it is not compared against other runs.

        Two distinct failures are caught. The first is a model that carries no
        information: it emits a near-constant probability, the F1-optimal
        threshold becomes 'label everything positive', and the confusion matrix
        shows perfect recall with a full complement of false positives. ROC-AUC
        near 0.5 gives that one away.

        The second is a model that ranks well but whose tuned threshold sits
        outside the range of its predictions, so every sample is labelled the
        same way. ROC-AUC is then high and healthy - 0.90 has been observed - yet
        F1, CSI, precision and recall are all zero. Keying on ROC-AUC alone
        misses it entirely, which is why the confusion matrix is checked in its
        own right.

        Parameters
        ----------
        y_pred: np.array
            The predicted probabilities.
        roc: float
            The ROC-AUC on this split.
        tp, tn, fp, fn: int
            The confusion matrix entries.
        period_name: str
            The split name, for the log message.

        Returns
        -------
        bool
            True when the predictions carry no usable ranking information.
        """
        spread = float(np.nanmax(y_pred) - np.nanmin(y_pred)) if y_pred.size else 0.0
        no_ranking = not np.isfinite(roc) or abs(roc - 0.5) < 0.01
        all_one_class = (tn + fp == 0) or (tp + fn == 0) or (tn == 0) or (fp + tp == 0)
        constant_output = all_one_class and spread < 1e-6

        # The threshold can collapse onto a single class while the model still
        # ranks perfectly well. With the positive class weight divided down, the
        # F1 optimum can sit above every predicted probability, so nothing is
        # called positive (tp = fp = 0), or below all of them, so everything is
        # (tn = fn = 0). ROC-AUC stays high and the checks above pass, yet every
        # threshold-derived column is trivial - which is what this flag exists to
        # mark. Not gated on the spread: that is precisely the case it misses.
        one_sided_threshold = (tp + fp == 0) or (tn + fn == 0)

        degenerate = no_ranking or constant_output or one_sided_threshold

        if no_ranking or constant_output:
            logger.warning(
                "Split '%s': the model produces no usable ranking (ROC-AUC=%.4f, "
                "prediction spread=%.3g). The confusion matrix above reflects the "
                "F1-optimal threshold applied to a near-constant output, not model "
                "skill - do not compare those columns across runs.",
                period_name, roc, spread)
        elif one_sided_threshold:
            side = 'negative' if tp + fp == 0 else 'positive'
            logger.warning(
                "Split '%s': the model ranks (ROC-AUC=%.4f) but the tuned "
                "threshold labels every sample %s (TP=%d, FP=%d, TN=%d, FN=%d). "
                "The threshold-derived columns (F1, CSI, precision, recall) are "
                "trivial by construction and must not be compared across runs; "
                "the ranking columns are still valid. A class weighting that "
                "favours the negatives is the usual cause.",
                period_name, roc, side, tp, fp, tn, fn)

        return degenerate

    def _assess_model(self, x, y, period_name, df_res):
        """
        Assess the model on a single period.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        logger.info("\nSplit: %s", period_name)

        df_tmp = pd.DataFrame(columns=df_res.columns)
        df_tmp['split'] = [period_name]

        # Compute the scores
        if self.target_type == 'occurrence':
            # Derive the class labels from the probabilities using the tuned
            # decision threshold (default 0.5, i.e. equivalent to predict()).
            y_pred_prob = self.model.predict_proba(x)[:, 1]
            y_pred = (y_pred_prob >= self.probability_threshold).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
            print_classic_scores(tp, tn, fp, fn)
            store_classic_scores(tp, tn, fp, fn, df_tmp)
            roc = assess_roc_auc(y, y_pred_prob)
            df_tmp['ROC_AUC'] = [roc]
            pr_auc, base_rate, pr_lift = assess_pr_auc(y, y_pred_prob)
            df_tmp['PR_AUC'] = [pr_auc]
            df_tmp['base_rate'] = [base_rate]
            df_tmp['PR_AUC_lift'] = [pr_lift]
            for name, value in assess_recall_at_budget(y, y_pred_prob).items():
                df_tmp[name] = [value]
            df_tmp['degenerate'] = [self._flag_degenerate_predictions(
                y_pred_prob, roc, tp, tn, fp, fn, period_name)]
        else:
            y_pred = self.model.predict(x)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            logger.info("RMSE: %s", rmse)
            df_tmp['RMSE'] = [rmse]
        logger.info("----------------------------------------")

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
        if self.df is not None:
            tag_data = (pickle.dumps(feature_files) + pickle.dumps(self.df.shape) +
                        pickle.dumps(self.df.columns) + pickle.dumps(self.df.iloc[0]) +
                        pickle.dumps(self.features))
        else:
            tag_data = pickle.dumps(feature_files) + pickle.dumps(self.features)

        df_hashed_name = f'data_{hashlib.md5(tag_data).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / df_hashed_name
        return tmp_filename

    def _define_potential_features(self, events_columns):
        self.tabular_features = {}

        if self.options.use_event_attributes:
            if self.options.event_method=='simple' or 'e_date' in events_columns:
                self.tabular_features['event'] = []
                if 'p_5min_q' in events_columns:
                    self.tabular_features['event'] += [
                        'p_5min_q', 'p_10min_q',
                        'p_20min_q', 'p_30min_q']

                self.tabular_features['event'] += [
                    'p_1h_q', 'p_2h_q', 'p_4h_q', 'p_6h_q',
                    'p_12h_q', 'p_24h_q', 'p_48h_q', 'p_72h_q',
                    'api_q', 'nb_contracts']
            else:
                self.tabular_features['event'] = [
                    'i_max_q', 'p_sum_q', 'duration', 'i_mean_q',
                    'api_q', 'nb_contracts']

            if getattr(self.options, 'use_poisson_head', False):
                # nb_contracts is the exposure offset; using it as a predictor
                # too would be double-use.
                self.tabular_features['event'].remove('nb_contracts')

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

