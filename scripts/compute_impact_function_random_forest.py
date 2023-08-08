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
from sklearn.model_selection import train_test_split
from pathlib import Path

LABEL_EVENT_FILE = 'original_w_prior_pluvial'


def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int)

    args = parser.parse_args()
    print("config: ", args.config)

    config = Config()
    tmp_dir = config.get('TMP_DIR')

    # Basic configuration - select hyperparameters and types of features
    max_depth = 6
    use_events_attributes = True
    use_swf_attributes = True
    use_terrain_attributes = True
    use_flowacc_attributes = True

    # Basic configuration - select features
    features_events = [
        'i_max_q', 'p_sum_q', 'e_tot', 'i_mean_q', 'apireg_q'
    ]
    features_swf = [
        # 'area_low', 'area_med', 'area_high',
        'area_exposed'
    ]
    features_terrain = [
        'dem_025m_slope_median', 'dem_010m_slope_min', 'dem_010m_curv_plan_std'
    ]
    features_flowacc = [
        'dem_010m_flowacc_norivers_median'
    ]

    # Configuration-specific changes
    if args.config == 1:
        features_swf = [
            'area_low', 'area_med', 'area_high', 'area_exposed'
        ]
    elif args.config == 2:
        features_swf = [
            'area_exposed'
        ]
    elif args.config == 3:
        max_depth = 8
    elif args.config == 4:
        max_depth = 10
    elif args.config == 5:
        max_depth = 20

    # Create list of static files
    static_files = []
    if use_swf_attributes:
        static_files.append(config.get('CSV_FILE_SWF'))
    if use_terrain_attributes:
        static_files.append(config.get('CSV_FILE_TERRAIN'))
    if use_flowacc_attributes:
        static_files.append(config.get('CSV_FILE_FLOWACC'))

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

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create unqiue hash for the data dataframe
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

    x = df[features].to_numpy()
    y = df['target'].to_numpy().astype(int)

    # Remove lines with NaN values
    x_nan = np.argwhere(np.isnan(x))
    rows_with_nan = np.unique(x_nan[:, 0])
    print(f"Removing {len(rows_with_nan)} rows with NaN values")
    x = np.delete(x, rows_with_nan, axis=0)
    y = np.delete(y, rows_with_nan, axis=0)

    assert len(np.argwhere(np.isnan(x))) == 0, f"NaN values in features: {features}"

    # Split the sample into training and test sets
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.3, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=42)

    # Class weights
    weights = len(y_train) / (2 * np.bincount(y_train))
    class_weight = {0: weights[0], 1: weights[1] / 16}

    tag_model = pickle.dumps(static_files) + pickle.dumps(events_filename) + \
                pickle.dumps(features) + pickle.dumps(class_weight) + \
                pickle.dumps(max_depth)
    model_hashed_name = f'rf_model_{hashlib.md5(tag_model).hexdigest()}.pickle'
    hash = hashlib.md5(tag_model).hexdigest()
    tmp_file = Path(tmp_dir) / model_hashed_name

    if tmp_file.exists():
        print(f"Loading model from {tmp_file}")
        rf = pickle.load(open(tmp_file, 'rb'))
        assess_random_forest(rf, x_train, y_train, 'Train period')
        assess_random_forest(rf, x_valid, y_valid, 'Validation period')
        assess_random_forest(rf, x_test, y_test, 'Test period')

    else:
        print(f"Training model and saving to {tmp_file}")
        rf = train_random_forest(x_train, y_train, class_weight, max_depth)
        assess_random_forest(rf, x_train, y_train, 'Train period')
        assess_random_forest(rf, x_valid, y_valid, 'Validation period')
        assess_random_forest(rf, x_test, y_test, 'Test period')
        pickle.dump(rf, open(tmp_file, "wb"))

    # Feature importance based on mean decrease in impurity
    print("Feature importance based on mean decrease in impurity")
    importances = rf.feature_importances_

    fig_filename = f'feature_importance_mdi_{args.config}_{hash}.pdf'
    plot_random_forest_feature_importance(rf, features, importances, fig_filename,
                                          dir_output=config.get('OUTPUT_DIR'))


def train_random_forest(x_train, y_train, class_weight='balanced', max_depth=10):
    print(f"Random forest with class weight: {class_weight}")
    rf = RandomForestClassifier(
        max_depth=max_depth, class_weight=class_weight, random_state=42,
        criterion='gini', n_jobs=20
    )
    rf.fit(x_train, y_train)

    return rf


def assess_random_forest(rf, x, y, title='Test period'):
    y_pred = rf.predict(x)
    y_pred_prob = rf.predict_proba(x)

    print(f"Random forest - {title}")

    # Compute the scores
    tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    assess_roc_auc(y, y_pred_prob[:, 1])


if __name__ == '__main__':
    main()
