"""
Score one or more prediction grids against a damages grid, and emit the rows
that go into the cross-model comparison table.

WHY THIS EXISTS. `09 Independent predictions/GVZ/Scores gvz.xlsx` compares
every model on the independent period 2023-2024: LR classic and simple, RF in
several versions, fifteen ANN variants - and no CNN row at all, although the
CNN is the model the whole of `06 CNN v2` is about. The grids needed to fill
that gap are already on disk, so the row costs a re-score and no prediction.

`assess_aggregation.py` produced the existing blocks, but it scores ONE
hard-coded file at ONE threshold under ONE relaxation setting, and prints.
Filling five seeds plus an ensemble across three aggregations and two
relaxation settings that way is 36 hand edits. This script takes the grids as
arguments, loops the settings, and writes a CSV whose columns are the
spreadsheet's, in the spreadsheet's order.

It also reports what a confusion matrix at a fixed threshold cannot: the
threshold-free metrics, the best operating point, and the recall ceiling - the
share of observed positives carrying any prediction at all, which is set by
the catalogue rather than the model and is the number `12 Catalogue at
inference/` showed dwarfs every model effect.

TWO THINGS IT DOES THAT THE OLD SCRIPT DID NOT.

1. It reopens the datasets for every pass. `prepare_full_domain_assessment`
   modifies `ds_pred['predict']` in place, because `.to_numpy()` on a loaded
   DataArray returns the array itself. Scoring one open dataset under several
   settings therefore scores the later ones against already-moved predictions.
   Measured on a real grid: the sum of `predict` changes after one pass.
2. It never mixes the two relaxation flags. `assess_aggregation.py` drove
   `ignore_removed` and `relax_days` from a single constant, which is why the
   spreadsheet's block labels only ever appear as both-False or both-True.
   They answer different questions and are separate arguments here.

Typical use - the CNN row for the GVZ table, five seeds and their ensemble:

    python scripts/impact_functions/score_prediction_grids.py \
        --damages "...\\09 Independent predictions\\GVZ\\damages_gvz_2023_2024.nc" \
        --grids "...\\12 Catalogue at inference\\predictions\\*_step1h_2023-2024.nc" \
        --ensemble --model-name "CNN simple hourly (anchor m05, step1h)" \
        --output "...\\09 Independent predictions\\GVZ\\scores_cnn.csv"

Add `--aggregations zip catchments` to fill the other two blocks; they need
the shapefiles and take a few minutes each.
"""
import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import precision_recall_curve

from swafi.config import Config
from swafi.utils.aggregation import aggregate_by_regions, build_region_map
from swafi.utils.logging_setup import setup_logging
from swafi.utils.verification import (assess_pr_auc, assess_recall_at_budget,
                                      assess_roc_auc,
                                      compute_confusion_matrix,
                                      compute_score_binary,
                                      prepare_full_domain_assessment)

logger = logging.getLogger(__name__)

config = Config()

# The shapefiles behind the spreadsheet's two coarser blocks. Kept as the
# defaults assess_aggregation.py used, overridable so the script does not
# depend on one machine's layout.
DEFAULT_SHAPEFILES = {
    'zip': R"C:\Data\Data\GIS\Administration\CH_zip_codes\AMTOVZ_ZIP.shp",
    'catchments': R"C:\Data\Projects\2024 SWF\Data\GIS\Catchments\ezgg_40km2.shp",
}

# The spreadsheet's metric column order. Keeping it means a row can be pasted
# in without rearranging, which is the point of the script.
SPREADSHEET_METRICS = [
    ('SEDI', 'SEDI'),
    ('False alarm rate (F)', 'F'),
    ('False alarm ratio (FAR)', 'FAR'),
    ('Hit rate (H)', 'H'),
    ('Critical Success Index (CSI)', 'CSI'),
    ('Bias', 'bias'),
    ('Accuracy', 'accuracy'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('F1', 'f1'),
]


def resolve_grids(patterns):
    """Expand the grid patterns to a sorted, de-duplicated list of files."""
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            candidate = Path(pattern)
            if candidate.exists():
                matched = [str(candidate)]
            else:
                raise FileNotFoundError(
                    f"No prediction grid matches '{pattern}'.")
        paths.extend(matched)

    # dict.fromkeys rather than set(): a stable order keeps the seeds in the
    # order their file names imply, which is the order the table wants.
    unique = list(dict.fromkeys(paths))
    logger.info("Scoring %s prediction grid(s).", len(unique))
    for path in unique:
        logger.info("  %s", Path(path).name)

    return unique


def build_ensemble(paths, out_path):
    """Average the member grids into one, the way seed ensembling scores it.

    The mean of the member probabilities, not a vote: that is what
    `06 Seed ensembling/` measured, and it beat the mean member by +7.2% on the
    pinned catalogue - a larger effect than any single design choice this
    project has priced.

    NaN handling is deliberately strict: a cell that is NaN in any member is
    NaN in the ensemble, so the ensemble is scored on the cells every member
    could speak to. Averaging over whichever members happen to cover a cell
    would make the ensemble's sample differ from its members'.
    """
    logger.info("Building the %s-member ensemble.", len(paths))
    # A running sum, not a stack: five members of this grid are 1.3 GB stacked
    # and NaN propagates through addition on its own, which is the behaviour
    # wanted here anyway.
    total = None
    template = None
    for path in paths:
        ds = xr.open_dataset(path)
        arr = ds['predict'].to_numpy().astype(np.float64)
        if total is None:
            total = arr
            template = ds[['predict']].copy(deep=True)
        elif arr.shape != total.shape:
            raise ValueError(
                f"{Path(path).name} has shape {arr.shape}, expected "
                f"{total.shape}: the members are not on one grid.")
        else:
            total += arr
        ds.close()

    template['predict'][:] = (total / len(paths)).astype(np.float32)
    template.to_netcdf(out_path)
    template.close()
    logger.info("Ensemble written to %s", out_path)

    return out_path


def recall_ceiling(y_true, y_pred):
    """The share of observed positives that carry any prediction at all.

    Not a model score: it is bounded by the catalogue, since a cell-day that
    never became a candidate carries no prediction at any threshold. It is
    identical across seeds within a catalogue, which is what proves it is the
    catalogue. Reported because every metric below is bounded by it, and CSI in
    particular is bounded above by recall, so the two compare in kind.
    """
    n_pos = int((y_true > 0).sum())
    if n_pos == 0:
        return np.nan, 0, 0
    unreachable = int(((y_true > 0) & (y_pred <= 0)).sum())

    return 1 - unreachable / n_pos, unreachable, n_pos


def best_operating_points(y_true, y_pred):
    """The best F1 and CSI over ALL thresholds, computed exactly.

    The fixed 0.5 the spreadsheet uses is not comparable across models
    calibrated to different positive rates - the five CNN seeds here span a
    factor of 2.9 in CSI at 0.5 while spanning 1.14 in best CSI, so most of
    that spread is calibration and not skill. The best-anywhere value is the
    threshold-free counterpart.

    This reads the full precision-recall curve rather than sweeping a grid of
    candidate thresholds. The first version swept 200 quantiles of the
    non-zero predictions and every optimum landed on the grid's top edge - on
    a problem with 329 positives in 1.29 M samples the optimum sits in the top
    few hundred predictions, far above the 99.5th percentile, so a quantile
    grid truncates it and reports different models as tied. The curve has one
    point per distinct threshold, so there is nothing left to truncate.

    CSI is derived from the same curve, exactly:
        1/P + 1/R - 1 = (TP+FP)/TP + (TP+FN)/TP - 1 = (TP+FP+FN)/TP = 1/CSI

    Returns
    -------
    dict
        Best F1 and CSI with the threshold each is attained at.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # precision_recall_curve appends a final (precision=1, recall=0) point
    # that no threshold produces, so it is dropped before matching up.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        csi = np.where((precision > 0) & (recall > 0),
                       1.0 / (1.0 / precision + 1.0 / recall - 1.0), 0.0)

    out = {}
    for name, values in (('F1', f1), ('CSI', csi)):
        if values.size == 0:
            out[f'best {name}'] = np.nan
            out[f'best {name} threshold'] = np.nan
            continue
        i = int(np.nanargmax(values))
        out[f'best {name}'] = float(values[i])
        out[f'best {name} threshold'] = float(thresholds[i])

    return out


def score_one(grid_path, damages_path, damages_reader, aggregation,
              shapefiles, ignore_removed, relax_days, prob_threshold,
              region_cache):
    """Score one grid under one setting, returning one table row.

    The datasets are opened here and closed before returning, which is what
    keeps the in-place modification in `prepare_full_domain_assessment` from
    leaking into the next setting.
    """
    ds_pred = xr.open_dataset(grid_path)
    ds_damages = damages_reader(damages_path)

    try:
        if aggregation == 'pixel':
            y_true, y_pred = prepare_full_domain_assessment(
                ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
        else:
            shapefile = shapefiles[aggregation]
            if not Path(shapefile).exists():
                raise FileNotFoundError(
                    f"Aggregation shapefile {shapefile} not found.")
            if aggregation not in region_cache:
                # The spatial join is the slow part and the grid geometry is
                # the same for every member, so it is done once per shapefile.
                region_cache[aggregation] = build_region_map(ds_pred, shapefile)
            region_map, region_ids, _ = region_cache[aggregation]
            if len(region_ids) == 0:
                raise ValueError(
                    "Aggregation shapefile did not overlap the grid at all.")
            y_true_reg, y_pred_reg = aggregate_by_regions(
                ds_pred, ds_damages, region_map, region_ids,
                ignore_removed=ignore_removed, relax_days=relax_days)
            y_true, y_pred = y_true_reg.ravel(), y_pred_reg.ravel()
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true, y_pred = y_true[mask], y_pred[mask]
    finally:
        ds_pred.close()
        ds_damages.close()

    if y_pred.size == 0:
        raise ValueError("No valid sample remains after masking.")

    y_true_bin = (y_true > 0).astype(int)
    ceiling, unreachable, n_pos = recall_ceiling(y_true_bin, y_pred)

    tp, tn, fp, fn = compute_confusion_matrix(
        y_true_bin, (y_pred >= prob_threshold).astype(int))

    row = {
        'samples': y_pred.size,
        'positives': n_pos,
        'base rate': n_pos / y_pred.size,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }
    for label, metric in SPREADSHEET_METRICS:
        row[label] = compute_score_binary(metric, tp, tn, fp, fn)

    # Threshold-free, and the ceiling every one of them sits under.
    ap, _, lift = assess_pr_auc(y_true_bin, y_pred)
    row['PR-AUC'] = ap
    row['PR-AUC lift'] = lift
    row['ROC-AUC'] = assess_roc_auc(y_true_bin, y_pred)
    row.update(best_operating_points(y_true_bin, y_pred))
    # Recall at a fixed alarm budget: the operational question, and the one
    # metric here that needs neither a threshold nor a calibration.
    row.update(assess_recall_at_budget(y_true_bin, y_pred))

    row['no prediction'] = unreachable
    row['recall ceiling'] = ceiling

    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--damages', required=True,
                        help="the damages netCDF for the test period")
    parser.add_argument('--grids', required=True, nargs='+', metavar='PATTERN',
                        help="prediction grids; globs are expanded")
    parser.add_argument('--dataset', default='gvz', choices=['gvz', 'mobiliar'],
                        help="the damage dataset the grids were predicted for")
    parser.add_argument('--model-name', default=None,
                        help="the name for the table's model column; the file "
                             "name is used when this is not given")
    parser.add_argument('--ensemble', action='store_true',
                        help="also score the mean of the given grids")
    parser.add_argument('--aggregations', nargs='+', default=['pixel'],
                        choices=['pixel', 'zip', 'catchments'],
                        help="the spatial units to score over")
    parser.add_argument('--zip-shapefile', default=DEFAULT_SHAPEFILES['zip'])
    parser.add_argument('--catchments-shapefile',
                        default=DEFAULT_SHAPEFILES['catchments'])
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="the probability threshold for the confusion "
                             "matrix (the spreadsheet's rows are 0.5)")
    parser.add_argument('--relax-days', nargs='+', type=int, default=[0, 1],
                        choices=[0, 1],
                        help="which relaxation settings to score; both by "
                             "default, so each spreadsheet block is filled in "
                             "one pass")
    parser.add_argument('--ignore-removed', nargs='+', type=int, default=None,
                        choices=[0, 1],
                        help="whether to drop the cell-days whose claim the "
                             "category selection removed; follows "
                             "--relax-days when not given, which is what the "
                             "existing table did")
    parser.add_argument('--output', default=None,
                        help="where to write the CSV; defaults beside the "
                             "damages file")
    args = parser.parse_args()

    setup_logging(script_name='score_prediction_grids')

    damages_path = Path(args.damages)
    if not damages_path.exists():
        raise FileNotFoundError(f"Damages file {damages_path} does not exist.")

    # Imported here so the script can be read without the damages classes'
    # dependencies, and so --help works anywhere.
    if args.dataset == 'mobiliar':
        from swafi.damages_mobiliar import DamagesMobiliar as DamagesClass
    else:
        from swafi.damages_gvz import DamagesGvz as DamagesClass

    def damages_reader(path):
        damages = DamagesClass(year_start=config.get('YEAR_START_TEST'),
                               year_end=config.get('YEAR_END_TEST'))
        return damages.from_nc_file(path)

    shapefiles = {'zip': args.zip_shapefile,
                  'catchments': args.catchments_shapefile}

    grids = resolve_grids(args.grids)
    labelled = [(Path(p).stem, p) for p in grids]

    if args.ensemble:
        if len(grids) < 2:
            raise ValueError("--ensemble needs at least two grids.")
        ensemble_path = damages_path.parent / (
            f"ensemble_{len(grids)}members_{damages_path.stem}.nc")
        build_ensemble(grids, ensemble_path)
        labelled.append((f"ENSEMBLE ({len(grids)} members)",
                         str(ensemble_path)))

    relax_settings = sorted(set(args.relax_days))
    if args.ignore_removed is None:
        # Follow relax_days, which is what the existing table did - its block
        # labels only ever read both-False or both-True.
        removed_settings = {r: r for r in relax_settings}
    else:
        removed_settings = {r: v for r in relax_settings
                            for v in sorted(set(args.ignore_removed))}

    region_cache = {}
    rows = []
    for aggregation in args.aggregations:
        for relax in relax_settings:
            ignore_removed = bool(removed_settings[relax])
            for name, path in labelled:
                logger.info("Scoring %s | %s | relax_days=%s "
                            "ignore_removed=%s", name, aggregation,
                            bool(relax), ignore_removed)
                row = score_one(
                    path, damages_path, damages_reader, aggregation,
                    shapefiles, ignore_removed, bool(relax), args.threshold,
                    region_cache)
                rows.append({
                    'model': args.model_name or name,
                    'grid': Path(path).name,
                    'aggregation': aggregation,
                    'relax_days': bool(relax),
                    'ignore_removed': ignore_removed,
                    'threshold': args.threshold,
                    **row,
                })

    df = pd.DataFrame(rows)

    output = Path(args.output) if args.output else (
        damages_path.parent / f"scores_{damages_path.stem}.csv")
    df.to_csv(output, index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 50)
    print()
    print("=" * 76)
    print("SCORES")
    print("=" * 76)
    show = ['grid', 'aggregation', 'relax_days', 'TP', 'FP', 'FN',
            'Critical Success Index (CSI)', 'F1', 'PR-AUC', 'ROC-AUC',
            'best F1', 'best CSI', 'no prediction', 'recall ceiling']
    print(df[[c for c in show if c in df.columns]].to_string(index=False))
    print()
    print("HOW TO READ THIS")
    print("-" * 76)
    print("  recall ceiling is the share of observed positives carrying any")
    print("  prediction at all. It is set by the catalogue, not the model, so")
    print("  it is identical across seeds - and every metric above sits under")
    print("  it. CSI is bounded above by recall, so best CSI against the")
    print("  ceiling is the one comparison of a score with its headroom that")
    print("  is legitimate in kind.")
    print()
    print("  The TP/FP/FN columns are taken at the fixed threshold, which is")
    print("  NOT comparable between models calibrated to different positive")
    print("  rates. Use best F1 / best CSI / PR-AUC to compare models, and the")
    print("  fixed threshold only to fill the existing table.")
    print()
    print(f"  Written to {output}")


if __name__ == '__main__':
    main()
