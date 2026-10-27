"""
Test script for loading and evaluating different pre-trained models.
"""
import logging
import numpy as np
import xarray as xr
from pathlib import Path

from swafi.config import Config
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.utils.aggregation import aggregate_by_regions, build_region_map
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, prepare_full_domain_assessment
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

AGGREGATIONS = [
    None,
    R"C:\Data\Data\GIS\Administration\CH_zip_codes\AMTOVZ_ZIP.shp",
    R"C:\Data\Projects\2024 SWF\Data\GIS\Catchments\ezgg_40km2.shp"
]

DO_ASSESS = True
DATASET = 'gvz'  # 'mobiliar' or 'gvz'
PREDICTION_FILE = R"C:\Data\Projects\2024 SWF\Analyses\09 Independent predictions\GVZ\pred_lr_gvz_classic_wd_20_2023-2024.nc"
DAMAGES_FILE = R"C:\Data\Projects\2024 SWF\Analyses\09 Independent predictions\GVZ\damages_gvz_2023_2024.nc"
# These answer different questions and used to be driven by one constant,
# which is why the score table's block labels only ever read both-False or
# both-True. RELAX_DAYS allows a prediction one day either side of the claim;
# IGNORE_REMOVED drops the cell-days whose claim the category selection
# removed. The other use_*_occurrence.py scripts keep them separate too.
RELAX_DAYS = False
IGNORE_REMOVED = False
THRESHOLD = 0.5

config = Config()


def assess(result_path, damages_reader, ignore_removed=False, relax_days=False,
           prob_threshold=0.5):
    """Score one prediction grid over each spatial aggregation.

    The datasets are REOPENED for every aggregation.
    `prepare_full_domain_assessment` modifies ds_pred['predict'] in place, so
    scoring one open dataset over several aggregations scored the later ones
    against predictions the pixel pass had already moved. This loop used to do
    exactly that - and closed ds_pred inside the loop as well, which only
    failed to raise because the array was already in memory.
    """
    for aggregation in AGGREGATIONS:
        print('Aggregation', aggregation)

        ds_pred = xr.open_dataset(result_path)
        ds_damages = damages_reader()

        if aggregation:
            if not Path(aggregation).exists():
                raise FileNotFoundError(f"Aggregation shapefile {aggregation} not found.")

            # Aggregate per polygon region
            region_map, region_ids, region_labels = build_region_map(ds_pred, aggregation)

            if len(region_ids) == 0:
                raise ValueError("Aggregation shapefile did not overlap the grid at all.")

            y_true_reg, y_pred_reg = aggregate_by_regions(
                ds_pred, ds_damages, region_map, region_ids,
                ignore_removed=ignore_removed, relax_days=relax_days
            )

            # Flatten and filter NaNs
            y_pred = y_pred_reg.ravel()
            y_true = y_true_reg.ravel()
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            if y_pred.size == 0:
                raise ValueError("After aggregation and filtering, no valid samples remain.")

        else:
            # Pixel-level assessment
            y_true, y_pred = prepare_full_domain_assessment(ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
            logger.info("Pixel-level assessment: %s samples.", y_pred.size)

        y_pred = (y_pred >= prob_threshold).astype(int)
        y_true = (y_true > 0).astype(int)
        tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
        print_classic_scores(tp, tn, fp, fn)
        logger.info("*************************************")
        ds_pred.close()
        ds_damages.close()


def get_damages(dataset):
    if dataset == 'mobiliar':
        damages = DamagesMobiliar(
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    elif dataset == 'gvz':
        damages = DamagesGvz(
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    else:
        raise ValueError(f"Unknown damage dataset: {dataset}")

    return damages


def get_damages_xr(dataset):
    damages = get_damages(dataset)
    claims_xr = damages.from_nc_file(DAMAGES_FILE)

    return claims_xr


def main():
    setup_logging(script_name='assess_aggregation')
    output_path = Path(PREDICTION_FILE)

    if not output_path.exists():
        raise (f"Prediction file {output_path} does not exist. "
               f"Please run the prediction script first.")

    assess(output_path, lambda: get_damages_xr(DATASET),
           ignore_removed=IGNORE_REMOVED, relax_days=RELAX_DAYS,
           prob_threshold=THRESHOLD)


if __name__ == '__main__':
    main()
