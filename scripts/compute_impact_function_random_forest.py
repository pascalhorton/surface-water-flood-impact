from enum import Enum, auto
from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, assess_roc_auc
from swafi.utils.plotting import plot_random_forest_feature_importance
import numpy as np
import pandas as pd
import argparse
import hashlib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from pathlib import Path


class Approach(Enum):
    ASSESSMENT = auto()
    GRID_SEARCH_CV = auto()
    AUTO = auto()


N_JOBS = 20
LABEL_EVENT_FILE = 'original_w_prior_pluvial'
APPROACH = Approach.GRID_SEARCH_CV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int, default=1,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    config = Config()
    tmp_dir = config.get('TMP_DIR')

    # Basic configuration - select hyperparameters and types of features
    max_depth = 10
    use_events_attributes = True
    use_swf_attributes = True
    use_terrain_attributes = True
    use_flowacc_attributes = True
    use_runoff_coeff_attributes = True
    use_land_cover_attributes = True

    # Basic configuration - select features
    features_events = [
        'i_max_q', 'p_sum_q', 'e_tot', 'i_mean_q', 'apireg_q',
        'i_max', 'p_sum', 'i_mean', 'apireg', 'nb_contracts'
    ]
    features_swf = [
        'area_low', 'area_med', 'area_high',
    ]
    features_terrain = [
        'dem_025m_slope_median', 'dem_010m_slope_min', 'dem_010m_curv_plan_std',

        'dem_100m_slope_mean', 'dem_100m_slope_median',
        'dem_050m_slope_mean', 'dem_050m_slope_median',
        'dem_025m_slope_mean',
        'dem_010m_slope_mean', 'dem_010m_slope_median',
    ]
    features_flowacc = [
        #'dem_010m_flowacc_norivers_median'

        'dem_010m_flowacc_norivers_max',
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
        'dem_100m_flowacc_norivers_median'
    ]
    features_runoff_coeff = [
        'runoff_coeff_max', 'runoff_coeff_mean'
    ]
    features_land_cover = [
        'land_cover_cat_1', 'land_cover_cat_2', 'land_cover_cat_3', 'land_cover_cat_4',
        'land_cover_cat_5', 'land_cover_cat_6', 'land_cover_cat_7', 'land_cover_cat_8',
        'land_cover_cat_9', 'land_cover_cat_10', 'land_cover_cat_11',
        'land_cover_cat_12'
    ]

    # Configuration-specific changes
    if args.config == 1:
        pass
    elif args.config == 2:
        pass
    elif args.config == 3:
        pass
    elif args.config == 4:
        pass
    elif args.config == 5:
        pass

    # Create list of static files
    static_files = []
    if use_swf_attributes:
        static_files.append(config.get('CSV_FILE_SWF'))
    if use_terrain_attributes:
        static_files.append(config.get('CSV_FILE_TERRAIN'))
    if use_flowacc_attributes:
        static_files.append(config.get('CSV_FILE_FLOWACC'))
    if use_runoff_coeff_attributes:
        static_files.append(config.get('CSV_FILE_RUNOFF_COEFF'))
    if use_land_cover_attributes:
        static_files.append(config.get('CSV_FILE_LAND_COVER'))

    # Create list of features
    features = []
    if use_events_attributes:
        features += features_events
    if use_swf_attributes:
        features += features_swf
    if use_terrain_attributes:
        features += features_terrain
    if use_flowacc_attributes:
        features += features_flowacc
    if use_runoff_coeff_attributes:
        features += features_runoff_coeff
    if use_land_cover_attributes:
        features += features_land_cover

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create unique hash for the data dataframe
    tag_data = pickle.dumps(static_files) + pickle.dumps(events_filename) + \
               pickle.dumps(features)
    df_hashed_name = f'rf_data_{hashlib.md5(tag_data).hexdigest()}.pickle'
    tmp_file = Path(tmp_dir) / df_hashed_name

    if tmp_file.exists():
        print(f"Loading data from {tmp_file}")
        df = pd.read_pickle(tmp_file)

    else:
        print(f"Creating dataframe and saving to {tmp_file}")
        df = events.events

        for f in static_files:
            df_new = pd.read_csv(f)

            # Filter out valid column names
            valid_columns = [col for col in features if col in df_new.columns] + ['cid']
            df_new = df_new[valid_columns]

            df = df.merge(df_new, on='cid', how='left')

        df.to_pickle(tmp_file)

    X = df[features].to_numpy()
    y = df['target'].to_numpy().astype(int)

    # Remove lines with NaN values
    X_nan = np.argwhere(np.isnan(X))
    rows_with_nan = np.unique(X_nan[:, 0])
    print(f"Removing {len(rows_with_nan)} rows with NaN values")
    X = np.delete(X, rows_with_nan, axis=0)
    y = np.delete(y, rows_with_nan, axis=0)

    assert len(np.argwhere(np.isnan(X))) == 0, f"NaN values in features: {features}"

    # Split the sample into training and test sets
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42)

    # Class weights
    weights = len(y_train) / (2 * np.bincount(y_train))
    class_weight = {0: weights[0], 1: weights[1] / 16}

    if APPROACH == Approach.ASSESSMENT:
        tag_model = pickle.dumps(static_files) + pickle.dumps(events_filename) + \
                    pickle.dumps(features) + pickle.dumps(class_weight) + \
                    pickle.dumps(max_depth)
        model_hashed_name = f'rf_model_{hashlib.md5(tag_model).hexdigest()}.pickle'
        hash = hashlib.md5(tag_model).hexdigest()
        tmp_file = Path(tmp_dir) / model_hashed_name

        if tmp_file.exists():
            print(f"Loading model from {tmp_file}")
            rf = pickle.load(open(tmp_file, 'rb'))
            assess_random_forest(rf, X_train, y_train, 'Train period')
            assess_random_forest(rf, X_valid, y_valid, 'Validation period')
            assess_random_forest(rf, X_test, y_test, 'Test period')

        else:
            print(f"Training model and saving to {tmp_file}")
            rf = train_random_forest(X_train, y_train, class_weight, max_depth)
            assess_random_forest(rf, X_train, y_train, 'Train period')
            assess_random_forest(rf, X_valid, y_valid, 'Validation period')
            assess_random_forest(rf, X_test, y_test, 'Test period')
            pickle.dump(rf, open(tmp_file, "wb"))

        # Feature importance based on mean decrease in impurity
        print("Feature importance based on mean decrease in impurity")
        importances = rf.feature_importances_

        fig_filename = f'feature_importance_mdi_{args.config}_{hash}.pdf'
        plot_random_forest_feature_importance(rf, features, importances, fig_filename,
                                              dir_output=config.get('OUTPUT_DIR'),
                                              n_features=30)

    elif APPROACH == Approach.GRID_SEARCH_CV:
        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   scoring='accuracy', cv=5, n_jobs=N_JOBS)

        # Perform grid search on training data
        grid_search.fit(X_train, y_train)

        # Print best parameters and corresponding accuracy score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Accuracy:", grid_search.best_score_)

        # Evaluate on test data using the best model
        best_rf = grid_search.best_estimator_
        test_accuracy = best_rf.score(X_test, y_test)
        print("Test Accuracy with Best Model:", test_accuracy)

        assess_random_forest(best_rf, X_train, y_train, 'Train period')
        assess_random_forest(best_rf, X_valid, y_valid, 'Validation period')
        assess_random_forest(best_rf, X_test, y_test, 'Test period')


def train_random_forest(X_train, y_train, class_weight='balanced', max_depth=10):
    print(f"Random forest with class weight: {class_weight}")
    rf = RandomForestClassifier(
        max_depth=max_depth, class_weight=class_weight, random_state=42,
        criterion='gini', n_jobs=20
    )
    rf.fit(X_train, y_train)

    return rf


def assess_random_forest(rf, X, y, title='Test period'):
    y_pred = rf.predict(X)
    y_pred_prob = rf.predict_proba(X)

    print(f"Random forest - {title}")

    # Compute the scores
    tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    assess_roc_auc(y, y_pred_prob[:, 1])


if __name__ == '__main__':
    main()
