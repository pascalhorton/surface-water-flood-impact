"""
Train a logistic regression model to predict the occurrence of damages.
"""

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.impact_lr import ImpactLogisticRegression


DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'

config = Config()


def main():
    events_filename = f'events_{DATASET}_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    lr = ImpactLogisticRegression(events)

    lr.load_features(['event', 'terrain', 'swf_map', 'runoff_coeff'])

    lr.split_sample()
    lr.normalize_features()
    lr.compute_balanced_class_weights()
    lr.compute_corrected_class_weights(weight_denominator=27)
    lr.fit()
    lr.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
