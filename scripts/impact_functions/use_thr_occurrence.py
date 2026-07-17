"""
Test script for loading and evaluating a threshold-based model.
"""
import logging
from pathlib import Path

from swafi.config import Config
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_thr import ImpactThresholds
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_damages_xr, get_events, create_prediction_dataset,
    GridPredictionWriter,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True

config = Config()


def main():
    setup_logging(script_name='use_thr_occurrence')
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_events(year_start, year_end, options.event_method,
                        precip_dataset=options.precip_dataset)

    for method in ['union', 'intersection']:
        output_path = (
            Path(config.get('OUTPUT_DIR'))
            / f'pred_thr_{options.dataset}_{options.event_method}'
              f'_{options.precip_dataset}_{method}_{year_start}-{year_end}.nc'
        )

        logger.info("Processing %s...", method)

        if output_path.exists():
            if DO_ASSESS:
                assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                       ignore_removed=True, relax_days=True)
            continue

        domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

        thr = ImpactThresholds(options)
        thr.tabular_features = {'event': ['i_max_q', 'p_sum_q']}
        thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method=method)

        writer = GridPredictionWriter(ds_pred, domain)
        writer.mask_outside_domain()

        cell_events = events[events['cid'].isin(writer.get_map_cids())]
        thr.set_events(cell_events)
        y_pred = thr.predict()
        assert len(y_pred) == len(cell_events)
        writer.write_events(cell_events['cid'], cell_events['i_max_date'], y_pred)

        ds_pred.to_netcdf(output_path)
        logger.info("Results saved to %s", output_path)
        ds_pred.close()

        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=True, relax_days=True)


if __name__ == '__main__':
    main()
