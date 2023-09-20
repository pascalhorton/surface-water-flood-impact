"""
Train a random forest model to predict the occurrence of damages.
"""

import argparse

from swafi.config import Config
from swafi.impact_rf import ImpactRandomForest
from swafi.events import load_events_from_pickle


LABEL_EVENT_FILE = 'original_w_prior_pluvial_damage_ratio'

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int, default=10,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    rf = ImpactRandomForest(events, target_type='damage_ratio', random_state=42)

    # Configuration-specific changes
    if args.config == 10:  # Manual configuration
        pass
    elif args.config == 11:
        pass
    elif args.config == 12:
        rf.select_nb_contracts_greater_or_equal_to(2)
    elif args.config == 13:
        rf.select_nb_contracts_greater_or_equal_to(5)
    elif args.config == 14:
        rf.select_nb_contracts_greater_or_equal_to(10)
    elif args.config == 15:
        rf.select_nb_contracts_greater_or_equal_to(20)
    elif args.config == 16:
        rf.select_nb_contracts_greater_or_equal_to(50)
    elif args.config == 17:
        rf.select_nb_contracts_greater_or_equal_to(100)

    rf.load_features(['event', 'terrain', 'swf_map', 'flowacc',
                      'land_cover', 'runoff_coeff'])

    rf.split_sample()
    rf.fit()
    rf.assess_model_on_all_periods()
    rf.plot_feature_importance(args.config, config.get('OUTPUT_DIR'))


if __name__ == '__main__':
    main()
