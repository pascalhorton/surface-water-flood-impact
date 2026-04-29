"""
Train a logistic regression model to predict the occurrence of damages.
"""
import logging

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_lr import ImpactLogisticRegression
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SAVE_MODEL = True

config = Config()

# Define the weight denominators. Optimal value is 30 for GVZ and 45 for Mobiliar.
weight_denominators = [5, 10, 20, 30, 40, 50]

# Enable to test different weight denominators
# weight_denominators = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]


def main():
    setup_logging(script_name='train_lr_occurrence')
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    options.print_options()

    for weight_denominator in weight_denominators:
        # Create the impact function
        lr = ImpactLogisticRegression(options, events)

        lr.select_features(options.replace_simple_features)
        lr.load_features(options.simple_feature_classes)

        lr.split_sample(valid_test_size=0.25, test_size=0)
        lr.normalize_features()
        lr.compute_balanced_class_weights()
        lr.compute_corrected_class_weights(weight_denominator=weight_denominator)
        lr.fit()

        tag_atts = options.get_attributes_tag()

        file_tag = f'lr_{tag_atts}_wd_{weight_denominator}'
        lr.assess_model_on_all_periods(save_results=True, file_tag=file_tag)

        if SAVE_MODEL:
            lr.save_model(dir_output=config.get('OUTPUT_DIR'),
                          base_name=f'model_lr_{options.dataset}_{options.event_file_label}_wd_{weight_denominator}')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))


if __name__ == '__main__':
    main()
