import core.damages
import core.events
import core.precipitation
from utils.forecast_verification import compute_confusion_matrix, compute_score
from utils.config import Config

CONFIG = Config()

TMP_DIR = CONFIG.get('TMP_DIR')
LABEL_RESULTING_FILE = 'original_pluvial'
THRESHOLD_I_MAX = 0.9
THRESHOLD_P_SUM = 0.98


def main():
    filename = f'events_with_target_values_{LABEL_RESULTING_FILE}.pickle'
    events = core.events.load_from_pickle(filename=filename)
    df = events.events

    # Count the number of events with and without damages
    events_with_damages = df[df['target'] > 0]
    events_without_damages = df[df['target'] == 0]
    print(f"Number of events with damages: {len(events_with_damages)}")
    print(f"Number of events without damages: {len(events_without_damages)}")

    # Apply the threshold method
    df['predict'] = 0
    df.loc[df['i_max_q'] >= THRESHOLD_I_MAX, 'predict'] = 1
    df.loc[df['p_sum_q'] >= THRESHOLD_P_SUM, 'predict'] = 1

    # Compute the confusion matrix
    tp, tn, fp, fn = compute_confusion_matrix(df)
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    print(f"SEDI: {compute_score('SEDI', tp, tn, fp, fn):.3f}")
    print(f"False alarm rate (F): {compute_score('F', tp, tn, fp, fn):.3f}")
    print(f"False alarm ratio (FAR): {compute_score('FAR', tp, tn, fp, fn):.3f}")
    print(f"Hit rate (H): {compute_score('H', tp, tn, fp, fn):.3f}")
    print(f"Bias: {compute_score('bias', tp, tn, fp, fn):.3f}")

    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")
    print(f"Precision: {tp / (tp + fp):.3f}")
    print(f"Recall: {tp / (tp + fn):.3f}")
    print(f"F1: {2 * tp / (2 * tp + fp + fn):.3f}")


if __name__ == '__main__':
    main()
