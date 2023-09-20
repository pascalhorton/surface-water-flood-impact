"""
Apply the thresholds used in the 2019 method to predict the occurrence of damages.
"""

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.impact_thr import ImpactThresholds


LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'

config = Config()


def main():
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    thr = ImpactThresholds(events)

    # Restrict the features to the ones used in the 2019 method
    thr.tabular_features = {'event': ['i_max_q', 'p_sum_q']}
    thr.load_features(['event'])

    thr.split_sample()
    thr.show_target_stats()

    print(f"Threshold 2019 method (union):")
    thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method='union')
    thr.assess_model_on_all_periods()

    print(f"Threshold 2019 method (intersection):")
    thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method='intersection')
    thr.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
