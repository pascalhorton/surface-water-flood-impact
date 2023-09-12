import argparse

from swafi.config import Config
from swafi.impact_rf import ImpactRandomForest
from swafi.events import load_events_from_pickle


LABEL_EVENT_FILE = 'original_w_prior_pluvial'

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI RF")
    parser.add_argument("config", help="Configuration", type=int, default=0,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    rf = ImpactRandomForest(events)

    # Configuration-specific changes
    if args.config == 0:  # Manual configuration
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.CSI
    elif args.config == 1:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.F1
    elif args.config == 2:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.F1_WEIGHTED
    elif args.config == 3:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.CSI
    elif args.config == 4:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.F1
    elif args.config == 5:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.F1_WEIGHTED
    elif args.config == 6:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.CSI
    elif args.config == 7:
        rf.optim_approach = rf.OptimApproach.MANUAL
        rf.optim_metric = rf.OptimMetric.CSI

    rf.load_features(['event', 'terrain', 'swf_map', 'flowacc',
                      'land_cover', 'runoff_coeff'])

    rf.split_sample()
    rf.compute_balanced_class_weights()
    rf.compute_corrected_class_weights(weight_denominator=27)
    rf.fit()
    rf.assess_model_on_all_periods()
    rf.plot_feature_importance(args.config, config.get('OUTPUT_DIR'))


if __name__ == '__main__':
    main()
