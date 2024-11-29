"""
Train a logistic regression model to predict the occurrence of damages.
"""

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_lr import ImpactLogisticRegression


config = Config()


def main():
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    lr = ImpactLogisticRegression(events, options)

    lr.select_features(options.replace_simple_features)
    lr.load_features(options.simple_feature_classes)

    lr.split_sample()
    lr.normalize_features()
    lr.compute_balanced_class_weights()
    lr.compute_corrected_class_weights(weight_denominator=30)
    lr.fit()

    tag_atts = options.get_attributes_tag()

    lr.assess_model_on_all_periods(save_results=True, file_tag=f'lr_{tag_atts}')


if __name__ == '__main__':
    main()
