"""
Train a random forest model to predict the occurrence of damages.
"""

from swafi.config import Config
from swafi.impact import Impact
from swafi.events import load_events_from_pickle


config = Config()


def main():
    # Load events
    events_filename = f'events_{options.dataset}_with_target_values_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    print("\nBenchmark model (always false):")
    bench = Impact(events, target_type='damage_ratio', random_state=None)
    bench.create_benchmark_model('always_false')
    bench.split_sample()
    bench.assess_model_on_all_periods()

    print("\nBenchmark model (always true):")
    bench = Impact(events, target_type='damage_ratio', random_state=None)
    bench.create_benchmark_model('always_true')
    bench.split_sample()
    bench.assess_model_on_all_periods()

    print("\nBenchmark model (random):")
    bench = Impact(events, target_type='damage_ratio', random_state=None)
    bench.create_benchmark_model('random')
    bench.split_sample()
    bench.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
