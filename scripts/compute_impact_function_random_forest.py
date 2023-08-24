from swafi.impact_rf import ImpactRandomForest

from swafi.events import load_events_from_pickle

from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, compute_score_binary
from swafi.utils.plotting import plot_random_forest_feature_importance

import argparse
import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score






LABEL_EVENT_FILE = 'original_w_prior_pluvial'




def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int, default=1,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    rf = ImpactRandomForest(events)


    # Configuration-specific changes
    if args.config == 1:
        rf.optim_approach = rf.OptimApproach.GRID_SEARCH_CV
        rf.optim_metric = rf.OptimMetric.F1
    elif args.config == 2:
        rf.optim_approach = rf.OptimApproach.GRID_SEARCH_CV
        rf.optim_metric = rf.OptimMetric.F1_WEIGHTED
    elif args.config == 3:
        rf.optim_approach = rf.OptimApproach.RANDOM_SEARCH_CV
        rf.optim_metric = rf.OptimMetric.F1
    elif args.config == 4:
        rf.optim_approach = rf.OptimApproach.RANDOM_SEARCH_CV
        rf.optim_metric = rf.OptimMetric.F1_WEIGHTED
    elif args.config == 5:
        rf.optim_approach = rf.OptimApproach.AUTO
        rf.optim_metric = rf.OptimMetric.F1
    elif args.config == 6:
        rf.optim_approach = rf.OptimApproach.AUTO
        rf.optim_metric = rf.OptimMetric.F1_WEIGHTED
    elif args.config == 7:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.CSI

    rf.load_features(['event', 'terrain', 'swf_map', 'flowacc',
                      'land_cover', 'runoff_coeff'])

    rf.split_sample()


    # Class weights
    weights = len(y_train) / (2 * np.bincount(y_train))
    class_weight = {0: weights[0], 1: weights[1] / WEIGHT_DENOMINATOR}

    # Scoring
    if OPTIM_METRIC == OptimMetric.F1:
        scoring = 'f1'
    elif OPTIM_METRIC == OptimMetric.F1_WEIGHTED:
        scoring = 'f1_weighted'
    elif OPTIM_METRIC == OptimMetric.CSI:
        scoring = 'csi'
    else:
        raise ValueError(f"Unknown optimizer metric: {OPTIM_METRIC}")

    if APPROACH == OptimApproach.MANUAL:
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

    elif APPROACH == OptimApproach.GRID_SEARCH_CV:
        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42, class_weight=class_weight)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   scoring=scoring, cv=5, n_jobs=N_JOBS)

        # Perform grid search on training data
        grid_search.fit(X_train, y_train)

        # Print best parameters and corresponding accuracy score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # Evaluate on test data using the best model
        best_rf = grid_search.best_estimator_
        test_accuracy = best_rf.score(X_test, y_test)
        print("Test Score with Best Model:", test_accuracy)

        assess_random_forest(best_rf, X_train, y_train, 'Train period')
        assess_random_forest(best_rf, X_valid, y_valid, 'Validation period')
        assess_random_forest(best_rf, X_test, y_test, 'Test period')

    elif APPROACH == OptimApproach.RANDOM_SEARCH_CV:
        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42, class_weight=class_weight)

        # Initialize RandomizedSearchCV
        rand_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                         scoring=scoring, cv=5, n_jobs=N_JOBS)

        # Perform grid search on training data
        rand_search.fit(X_train, y_train)

        # Print best parameters and corresponding accuracy score
        print("Best Parameters:", rand_search.best_params_)
        print("Best Score:", rand_search.best_score_)

        # Evaluate on test data using the best model
        best_rf = rand_search.best_estimator_
        test_accuracy = best_rf.score(X_test, y_test)
        print("Test Score with Best Model:", test_accuracy)

        assess_random_forest(best_rf, X_train, y_train, 'Train period')
        assess_random_forest(best_rf, X_valid, y_valid, 'Validation period')
        assess_random_forest(best_rf, X_test, y_test, 'Test period')

    elif APPROACH == OptimApproach.AUTO:
        # Define objective function for Optuna
        def objective(trial):
            weight_denominator = trial.suggest_int('weight_denominator', 1, 50)
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            max_features = trial.suggest_categorical('max_features',
                                                     [None, 'sqrt', 'log2'])

            class_weight = {0: weights[0], 1: weights[1] / weight_denominator}

            rf_classifier = RandomForestClassifier(
                class_weight=class_weight,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )

            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_valid)

            if OPTIM_METRIC == OptimMetric.F1:
                return f1_score(y_valid, y_pred)
            elif OPTIM_METRIC == OptimMetric.F1_WEIGHTED:
                return f1_score(y_valid, y_pred, sample_weight=class_weight)
            elif OPTIM_METRIC == OptimMetric.CSI:
                tp, tn, fp, fn = compute_confusion_matrix(y_valid, y_pred)
                csi = compute_score_binary('CSI', tp, tn, fp, fn)
                return csi
            else:
                raise ValueError(f"Unknown optimizer metric: {OPTIM_METRIC}")

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # Record the value for the last time
        study_file = Path(tmp_dir) / f'rf_study_{args.config}.pickle'
        pickle.dump(study, open(study_file, "wb"))

        # Print optimization results
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train and evaluate the best model on test data
        best_params = study.best_params
        best_rf = RandomForestClassifier(**best_params, random_state=42)
        best_rf.fit(X_train, y_train)
        y_pred = best_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy with Best Model:", test_accuracy)

        assess_random_forest(best_rf, X_train, y_train, 'Train period')
        assess_random_forest(best_rf, X_valid, y_valid, 'Validation period')
        assess_random_forest(best_rf, X_test, y_test, 'Test period')


def train_random_forest(X_train, y_train, class_weight='balanced', max_depth=10):
    print(f"Random forest with class weight: {class_weight}")
    rf = RandomForestClassifier(
        max_depth=max_depth, class_weight=class_weight, random_state=42,
        criterion='gini', n_jobs=N_JOBS
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
