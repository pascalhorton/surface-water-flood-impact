"""
Train a random forest model to predict the occurrence of damages.
"""

import argparse

from swafi.config import Config
from swafi.impact_rf import ImpactRandomForest
from swafi.events import load_events_from_pickle


DATASET = 'gvz'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int, default=0,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_{DATASET}_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    rf = ImpactRandomForest(events, target_type='occurrence', random_state=None)

    # Configuration-specific changes
    if args.config == 0:  # Manual configuration
        pass
    elif args.config == 1:
        pass
    elif args.config == 2:
        rf.select_nb_claims_greater_or_equal_to(2)
    elif args.config == 3:
        rf.select_nb_claims_greater_or_equal_to(3)
    elif args.config == 4:
        rf.select_nb_claims_greater_or_equal_to(4)
    elif args.config == 5:
        rf.select_nb_claims_greater_or_equal_to(2)
    elif args.config == 6:
        rf.select_nb_claims_greater_or_equal_to(3)
    elif args.config == 7:
        rf.select_nb_claims_greater_or_equal_to(4)

    rf.load_features(['event', 'terrain', 'swf_map', 'flowacc', 'twi',
                      'land_cover', 'runoff_coeff'])

    rf.split_sample()
    rf.compute_balanced_class_weights()
    rf.compute_corrected_class_weights(weight_denominator=27)
    rf.fit()
    rf.assess_model_on_all_periods()
    rf.plot_feature_importance(args.config, config.get('OUTPUT_DIR'))


if __name__ == '__main__':
    main()
