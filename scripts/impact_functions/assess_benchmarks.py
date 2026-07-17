"""
Train a random forest model to predict the occurrence of damages.
"""

import logging
from swafi.config import Config
from swafi.impact import Impact
from swafi.events import load_events_from_pickle
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.utils.logging_setup import setup_logging

config = Config()


def main():
    setup_logging(script_name='assess_benchmarks')
    logger = logging.getLogger(__name__)
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Load events
    events = load_events_from_pickle(filename=options.get_events_filename())
    events.check_precip_dataset(options.precip_dataset)

    # Create the impact function
    logger.info("Benchmark model (always false):")
    bench = Impact(events, options)
    bench.create_benchmark_model('always_false')
    bench.split_sample(valid_test_size=0.25, test_size=0)
    bench.assess_model_on_all_periods(save_results=True, file_tag='bench_false')

    logger.info("Benchmark model (always true):")
    bench = Impact(events, options)
    bench.create_benchmark_model('always_true')
    bench.split_sample(valid_test_size=0.25, test_size=0)
    bench.assess_model_on_all_periods(save_results=True, file_tag='bench_true')

    logger.info("Benchmark model (random):")
    bench = Impact(events, options)
    bench.create_benchmark_model('random')
    bench.split_sample(valid_test_size=0.25, test_size=0)
    bench.assess_model_on_all_periods(save_results=True, file_tag='bench_rand')


if __name__ == '__main__':
    main()
