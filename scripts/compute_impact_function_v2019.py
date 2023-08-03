import numpy as np

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores
from sklearn.model_selection import train_test_split

CONFIG = Config()

TMP_DIR = CONFIG.get('TMP_DIR')
LABEL_RESULTING_FILE = 'original_w_prior_pluvial'
THRESHOLD_I_MAX = 0.9
THRESHOLD_P_SUM = 0.98


def main():
    filename = f'events_with_target_values_{LABEL_RESULTING_FILE}.pickle'
    events = load_events_from_pickle(filename=filename)
    df = events.events

    # Count the number of events with and without damages
    events_with_damages = df[df['target'] > 0]
    events_without_damages = df[df['target'] == 0]
    print(f"Number of events with damages: {len(events_with_damages)}")
    print(f"Number of events without damages: {len(events_without_damages)}")

    # Extract the features and the target
    x = df[['i_max_q', 'p_sum_q']].to_numpy()
    y = df['target'].to_numpy()

    # Split the sample into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)

    # Apply the threshold method (union)
    y_pred = np.zeros(len(y_test))
    y_pred[x_test[:, 0] >= THRESHOLD_I_MAX] = 1
    y_pred[x_test[:, 1] >= THRESHOLD_P_SUM] = 1

    # Compute the confusion matrix
    print(f"Threshold 2019 method (union):")
    tp, tn, fp, fn = compute_confusion_matrix(y_test, y_pred)
    print_classic_scores(tp, tn, fp, fn)

    # Apply the threshold method (intersection)
    y_pred = np.zeros(len(y_test))
    y_pred[(x_test[:, 0] >= THRESHOLD_I_MAX) & (x_test[:, 1] >= THRESHOLD_P_SUM)] = 1

    # Compute the confusion matrix
    print(f"Threshold 2019 method (intersection):")
    tp, tn, fp, fn = compute_confusion_matrix(y_test, y_pred)
    print_classic_scores(tp, tn, fp, fn)


if __name__ == '__main__':
    main()
