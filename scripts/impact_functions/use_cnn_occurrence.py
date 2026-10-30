"""
Test script for loading and evaluating a pre-trained CNN model.

MODEL and DATASET below are defaults; --model and --dataset override them, so
a batch of models can be run without editing the file. Everything else about
the run - the precipitation product, the window, the anchor, the features - is
read off the saved model, which is what keeps inference from drifting from
training.

Note there is NO --event-file-label here and none is needed: inference builds
its test events from the training CATALOGUE named by EVENTS_PATH_SIMPLE_*
(plus its '_ref.pkl' for the quantile features), never from a linked sample.

Use --check-inputs first when submitting an array job: it resolves and tests
every input, prints the output path the run would write, and exits without
touching the GPU.
"""
import argparse
import logging
import keras
import random
import tensorflow as tf
import numpy as np
from pathlib import Path

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.impact_dl import WeightedBinaryCrossEntropy, CriticalSuccessIndex
from swafi.utils.logging_setup import setup_logging
from swafi.utils.event_extraction import detection_tag_in
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_inference_events, get_training_catalogue_path,
    resolve_precip_reference, filter_events_to_precip_domain,
    create_prediction_dataset, create_precipitation, ensure_precip_dataset,
    GridPredictionWriter, predict_events_in_chunks,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True

# Full-domain scoring settings. These were False/False while every other
# use_*_occurrence.py script used True/True, which made the CNN's full-domain
# numbers quietly incomparable with the baselines' - the CNN was scored under
# the harsher rule.
#
# RELAX_DAYS moves a neighbouring day's prediction onto the claim day when that
# neighbour carries no claim of its own. It is justified: claim-date noise is
# measured, twice (9.6% of GVZ claims over 2005-2022, and half the unreachable
# cell-days on 2023-2024). But it is the GENEROUS bound - each move turns a
# false alarm and a miss into a hit and a correct negative, while the placebo
# puts only ~47% of +/-1 day associations down to a real storm. assess() now
# logs the ceiling under BOTH settings, so the bracket is always visible and
# this constant no longer hides half the answer.
#
# IGNORE_REMOVED excludes cell-days whose claim was dropped by the category
# selection. Those are not clean negatives: there WAS water damage, of another
# category, so scoring a prediction there as a false alarm charges the model
# for being roughly right.
RELAX_DAYS = True
IGNORE_REMOVED = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_cnn_test_30.keras"
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'

# The event definition is read off the training catalogue named by the
# EVENTS_PATH_SIMPLE_* / EVENTS_PATH_CLASSIC config entry, which is the only
# record of how it was detected (see use_common.get_inference_events). Keeping
# a second copy of those settings here is what let inference drift from
# training. The '*_q' features are ranked against the training distribution,
# as the model was fitted; --no-training-reference re-estimates them on the
# test period instead, which needs no '_ref.pkl' but is not what the model
# saw.

config = Config()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=MODEL,
                        help="the .keras model to run inference with")
    parser.add_argument('--dataset', default=DATASET,
                        choices=['gvz', 'mobiliar'],
                        help="the damage dataset to predict for")
    parser.add_argument('--no-assess', action='store_true',
                        help="write the grid without scoring it")
    parser.add_argument('--no-training-reference', action='store_true',
                        help="re-estimate the '*_q' features on the test "
                             "period instead of ranking them against the "
                             "training distribution the model was fitted on")
    parser.add_argument('--check-inputs', action='store_true',
                        help="resolve and test every input, print the output "
                             "path, and exit; nothing is computed")
    return parser.parse_args()


def check_inputs(args):
    """Verify everything a run needs, and report what it would write.

    An array job that fails on a missing file after loading the precipitation
    has wasted an hour per task for something a path test answers in a second.
    Each line is printed with its own OK/MISSING, so a failure names the file
    rather than the stage.
    """
    problems = []

    def report(label, path, needed=True):
        exists = path is not None and Path(path).exists()
        state = 'OK     ' if exists else ('MISSING' if needed else 'absent ')
        print(f"  [{state}] {label:24} {path}")
        if needed and not exists:
            problems.append(f"{label}: {path}")
        return exists

    print("=" * 76)
    print("INPUTS")
    print("=" * 76)
    if not report('model', args.model):
        print()
        print("The model could not be read, so the rest cannot be resolved "
              "from it.")
        return 1

    cnn_model = keras.models.load_model(args.model)
    options = cnn_model.options
    options.dataset = args.dataset
    ensure_precip_dataset(options)

    print()
    print(f"  run name        {options.run_name}")
    print(f"  dataset         {options.dataset}")
    print(f"  event method    {options.event_method}")
    print(f"  precip dataset  {options.precip_dataset}")
    # event_ref_date IS the anchor (Impact.set_reference_date): 'auto' is the
    # calendar day, 'i_max_only' the storm peak. It is the whole difference
    # between the two configurations these runs compare, so it is printed
    # first - a check pass that did not show it could pass ten identical runs.
    for attr in ('event_ref_date', 'precip_window_size', 'precip_time_step',
                 'precip_hours_before', 'precip_hours_after',
                 'random_state'):
        if hasattr(options, attr):
            print(f"  {attr:20} {getattr(options, attr)}")
    print()

    # Every config lookup is guarded. A missing entry is exactly what this
    # pass exists to report: raising here would hide the other nine tasks'
    # problems behind a traceback for the first one.
    def from_config(key):
        try:
            return config.get(key)
        except (RuntimeError, KeyError) as err:
            print(f"  [MISSING] config entry {key}: {err}")
            problems.append(f"config entry {key}")
            return None

    catalogue = None
    key = ('EVENTS_PATH_SIMPLE_' if options.event_method == 'simple'
           else 'EVENTS_PATH_CLASSIC_') + (
        '5MIN' if options.precip_dataset == '5min' else 'HOURLY')
    raw = from_config(key)
    if raw is not None:
        catalogue = Path(raw)
        report(f'catalogue ({key})', catalogue)
        ref = catalogue.with_name(
            catalogue.name.replace('.parquet', '_ref.pkl'))
        report('training reference', ref,
               needed=not args.no_training_reference)

    out_dir = from_config('OUTPUT_DIR')
    out_dir = Path(out_dir) if out_dir else None
    if out_dir is not None:
        report('OUTPUT_DIR', out_dir)

    year_start = from_config('YEAR_START_TEST')
    year_end = from_config('YEAR_END_TEST')
    if catalogue is None or out_dir is None or year_start is None:
        print()
        print("MISSING:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    det_tag = detection_tag_in(catalogue) or '_detdefault'
    output_path = (
        out_dir
        / f'pred_cnn_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{options.run_name}{det_tag}'
          f'_{year_start}-{year_end}.nc'
    )
    print()
    print(f"  test period     {year_start}-{year_end}")
    print(f"  detection tag   {det_tag}")
    print(f"  would write     {output_path}")
    if output_path.exists():
        print("                  (EXISTS: the run would only re-score it)")

    print()
    if problems:
        print("MISSING:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("All inputs present.")
    return 0


def main():
    setup_logging(script_name='use_cnn_occurrence')
    args = parse_args()

    if args.check_inputs:
        raise SystemExit(check_inputs(args))

    cnn_model = keras.models.load_model(args.model)

    options = cnn_model.options
    options.dataset = args.dataset
    # Before print_options(): both it and is_ok() read the precipitation dataset.
    ensure_precip_dataset(options)
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    keras.backend.clear_session()
    if options.random_state is not None:
        random.seed(options.random_state)
        np.random.seed(options.random_state)
        tf.random.set_seed(options.random_state)
        keras.utils.set_random_seed(options.random_state)

    if cnn_model.model is None:
        cnn_model.build_model()

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_inference_events(
        options, year_start, year_end,
        use_training_reference=not args.no_training_reference)

    # Precipitation for CNN input, from the zarr store of the same dataset the
    # model was trained on.
    cpc = create_precipitation(options.precip_dataset, year_start, year_end)

    # The detection tag belongs in the NAME. The catalogue decides which
    # cell-days become candidates at all, so the same model against two
    # catalogues gives two different grids - and nothing inside a .nc records
    # which catalogue produced it. Without the tag the second run silently
    # reuses the first one's cached grid, which is the defect
    # `../12 Catalogue at inference/` was built to fix in its own script and
    # which was never carried back here. Six caches in this project have been
    # keyed on less than determines their contents; this would have been the
    # seventh.
    catalogue = get_training_catalogue_path(options.event_method,
                                            options.precip_dataset)
    det_tag = detection_tag_in(catalogue) or '_detdefault'
    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_cnn_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{options.run_name}{det_tag}'
          f'_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS and not args.no_assess:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=IGNORE_REMOVED, relax_days=RELAX_DAYS,
                   prob_threshold=0.5)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    cnn = ImpactCnn(options)
    cnn.set_model(cnn_model)
    cnn.set_precipitation(cpc)
    features = None
    if cnn.options.use_static_attributes or cnn.options.use_event_attributes:
        # Align the default event features with the loaded events: sub-hourly
        # features (5-min dataset) are only included when present in the events.
        cnn.update_potential_features(events.columns)
        cnn.select_features(cnn.options.replace_simple_features)
        features = cnn.get_all_features(cnn.options.simple_feature_classes)

    # None when the statistics stored in the model are used; the precipitation
    # is then cropped to the domain the model was trained on, and so are the
    # events (no external reference to cover a wider one).
    precip_stats, cropped = resolve_precip_reference(
        cnn_model, options, cnn.precipitation_hf)
    if cropped:
        events = filter_events_to_precip_domain(
            events, cnn.precipitation_hf, options)
    dg = cnn.get_data_generator_inference(
        events=events,
        features=features,
        exposure=contracts_number,
        precip_stats=precip_stats,
    )
    if precip_stats is not None:
        precip_stats.close()

    writer = GridPredictionWriter(ds_pred, domain)
    writer.mask_outside_domain()

    # NaN masks, in the same precedence as the former per-cell loop: cells
    # without events stay at 0; cells with events but no (or zero) exposure
    # are NaN.
    cids_events = set(events['cid'].unique()) & writer.get_map_cids()
    cids_exposure = set(
        contracts_number.loc[contracts_number['nb_contracts'] != 0, 'cid'])
    writer.fill_cells(cids_events - cids_exposure, np.nan)

    predict_cids = cids_events & cids_exposure
    event_cids = dg.event_props[:, 3].astype(np.int64)
    idxs = np.where(np.isin(
        event_cids, np.fromiter(predict_cids, dtype=np.int64)))[0]
    if len(idxs) > 0:
        y_pred = predict_events_in_chunks(cnn.model, dg, idxs, chunk_size=1024)
        # Poisson head: rate -> P(>=1) = 1 - exp(-rate); no-op otherwise.
        y_pred = cnn._predictions_to_proba(y_pred)
        writer.write_events(event_cids[idxs], dg.event_props[idxs, 0], y_pred)

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS and not args.no_assess:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=IGNORE_REMOVED, relax_days=RELAX_DAYS,
               prob_threshold=0.5)


if __name__ == '__main__':
    main()
