from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, assess_roc_auc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CONFIG = Config()

PICKLES_DIR = CONFIG.get('PICKLES_DIR')
LABEL_RESULTING_FILE = 'original_w_prior_pluvial'

# Possible features: 'e_start', 'e_end', 'e_tot', 'p_sum', 'p_sum_q', 'i_max',
# 'i_max_q', 'i_mean', 'i_mean_q', 'i_sd', 'i_sd_q', 'apireg', 'apireg_q'
FEATURES = ['i_max_q', 'p_sum_q', 'e_tot', 'i_mean_q', 'apireg_q']


def main():
    filename = f'events_with_target_values_{LABEL_RESULTING_FILE}.pickle'
    events = load_events_from_pickle(filename=filename)
    df = events.events

    x = df[FEATURES].to_numpy()
    y = df['target'].to_numpy().astype(int)

    assert len(np.argwhere(np.isnan(x))) == 0, f"NaN values in features: {FEATURES}"

    # Standardize the features
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Split the sample into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)

    # Default balanced class weight
    apply_logistic_regression(x_train, y_train, x_test, y_test, 'balanced')

    # Different class weights
    weights = len(y_train) / (2 * np.bincount(y_train))
    class_weight = {0: weights[0], 1: weights[1] / 2}
    apply_logistic_regression(x_train, y_train, x_test, y_test, class_weight)

    class_weight = {0: weights[0], 1: weights[1] / 4}
    apply_logistic_regression(x_train, y_train, x_test, y_test, class_weight)

    class_weight = {0: weights[0], 1: weights[1] / 8}
    apply_logistic_regression(x_train, y_train, x_test, y_test, class_weight)

    class_weight = {0: weights[0], 1: weights[1] / 16}
    apply_logistic_regression(x_train, y_train, x_test, y_test, class_weight)


def apply_logistic_regression(x_train, y_train, x_test, y_test, class_weight):
    print(f"Logistic regression with class weight: {class_weight}")
    clf = LogisticRegression(class_weight=class_weight).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_prob = clf.predict_proba(x_test)

    # Compute the scores
    tp, tn, fp, fn = compute_confusion_matrix(y_test, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    assess_roc_auc(y_test, y_pred_prob[:, 1])

    return clf


if __name__ == '__main__':
    main()
