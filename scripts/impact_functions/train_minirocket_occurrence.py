"""
Train a MiniRocket + ridge classifier on the at-cell precipitation series to
predict the occurrence of damages to buildings.

Diagnostic companion to the CNN model: MiniRocket features are computed from
the same single-pixel precipitation series the CNN sees, and a linear ridge
classifier is trained on (a) the MiniRocket features, (b) the tabular
features, and (c) both. Comparing the three answers whether the raw series
carries signal beyond the tabular event summaries.

The full dataset (millions of events at a prevalence below 0.2%) does not fit
in a dense feature matrix, so the negatives are subsampled (all positives are
kept). ROC-AUC and the ranking of the three variants are unaffected by that
subsampling; precision, CSI and F1 refer to the subsampled prevalence, which
is reported in the results, and are not comparable to the CNN scores.

Script-specific arguments (in addition to the usual CNN options):
    --minirocket-features N   Number of MiniRocket features (default 2520).
    --max-negatives N         Negatives kept per split (default 50000, 0 = all).
"""

import argparse
import logging
import random
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from swafi.config import Config
from swafi.events import load_events_from_pickle
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.logging_setup import setup_logging
from swafi.utils.minirocket import MiniRocket
from swafi.utils.verification import (
    compute_confusion_matrix, print_classic_scores, store_classic_scores,
    assess_roc_auc)

logger = logging.getLogger(__name__)

BATCH_SIZE_EXTRACT = 1024
ALPHAS = np.logspace(-1, 4, 6)

config = Config()


def main():
    script_args = _parse_script_args()
    setup_logging(script_name='train_minirocket_occurrence')
    options = ImpactCnnOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.use_precip, "MiniRocket needs the precipitation series."
    assert options.precip_window_size == options.precip_resolution, \
        "MiniRocket is univariate; use a single-pixel window."
    assert not options.use_dem, "The DEM channel is not used by MiniRocket."
    assert not options.use_poisson_head, \
        "The Poisson head does not apply to the ridge classifier."

    seed = options.random_state if options.random_state is not None else 42
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    logger.info("MiniRocket features: %d; negatives kept per split: %s",
                script_args.minirocket_features,
                script_args.max_negatives or 'all')

    if options.dataset == 'mobiliar':
        year_start = config.get('YEAR_START_MOBILIAR')
        year_end = config.get('YEAR_END_MOBILIAR')
    elif options.dataset == 'gvz':
        year_start = config.get('YEAR_START_GVZ')
        year_end = config.get('YEAR_END_GVZ')
    else:
        raise ValueError(f'Dataset {options.dataset} not recognized.')

    events = load_events_from_pickle(filename=options.get_events_filename())
    events.check_precip_dataset(options.precip_dataset)

    precip = CombiPrecip(year_start, year_end)

    # The CNN class is used here only as the data pipeline (event/claims
    # linkage, splits, per-event precipitation extraction).
    cnn = ImpactCnn(options, events)
    cnn.set_precipitation(precip)
    cnn.remove_events_without_precipitation_data()
    cnn.reduce_spatial_domain(options.precip_window_size)
    has_tabular = (options.use_static_attributes
                   or options.use_event_attributes)
    if has_tabular:
        cnn.select_features(options.replace_simple_features)
        cnn.load_features(options.simple_feature_classes)
    cnn.split_sample(valid_test_size=0.25, test_size=0)

    cnn._create_data_generator_train()
    cnn._create_data_generator_valid()
    splits = {'train': cnn.dg_train, 'valid': cnn.dg_val}
    if cnn.events_test is not None and len(cnn.events_test) > 0:
        cnn._create_data_generator_test()
        splits['test'] = cnn.dg_test

    series, static, targets, prevalence = {}, {}, {}, {}
    for name, dg in splits.items():
        idxs = _select_indices(dg.y, script_args.max_negatives, rng)
        logger.info("Extracting the precipitation series (%s: %d of %d events).",
                    name, len(idxs), len(dg.y))
        series[name], static[name], targets[name] = _materialize(dg, idxs)
        prevalence[name] = float(np.mean(targets[name] > 0))
        logger.info("%s: %d events, series length %d, prevalence %.3f%%",
                    name, len(targets[name]), series[name].shape[1],
                    100 * prevalence[name])

    logger.info("Fitting MiniRocket on the training series.")
    rocket = MiniRocket(num_features=script_args.minirocket_features,
                        random_state=seed)
    rocket.fit(series['train'])
    n_total = sum(len(t) for t in targets.values())
    logger.info("Transforming all splits (%.1f GB of features).",
                n_total * rocket.num_features * 4 / 1e9)
    rocket_feats = {name: rocket.transform(s) for name, s in series.items()}
    series.clear()

    variants = {'series': rocket_feats}
    if has_tabular:
        variants['static'] = static
        variants['series+static'] = {
            name: np.hstack([rocket_feats[name], static[name]])
            for name in splits}

    for variant, feats in variants.items():
        logger.info("=" * 60)
        logger.info("Variant: %s (%d features)",
                    variant, feats['train'].shape[1])

        scaler = StandardScaler().fit(feats['train'])
        scaled = {name: scaler.transform(f) for name, f in feats.items()}
        clf = _fit_ridge(scaled['train'], targets['train'],
                         scaled['valid'], targets['valid'])

        scores = {name: clf.decision_function(f) for name, f in scaled.items()}
        del scaled
        threshold = _find_best_f1_threshold(scores['valid'], targets['valid'])
        logger.info("Decision threshold from validation (F1): %.4f", threshold)

        df_res = pd.DataFrame(columns=['split'])
        for name in splits:
            logger.info("\nSplit: %s", name)
            y_obs = targets[name]
            y_cls = (scores[name] >= threshold).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_cls)
            print_classic_scores(tp, tn, fp, fn)
            df_tmp = pd.DataFrame(columns=df_res.columns)
            df_tmp['split'] = [name]
            store_classic_scores(tp, tn, fp, fn, df_tmp)
            df_tmp['ROC_AUC'] = [assess_roc_auc(y_obs, scores[name])]
            df_tmp['PR_AUC'] = [average_precision_score(y_obs, scores[name])]
            # Precision-based scores refer to this (subsampled) prevalence.
            df_tmp['prevalence'] = [prevalence[name]]
            df_res = pd.concat([df_res, df_tmp])
            logger.info("ROC AUC: %.4f, PR AUC: %.4f",
                        df_tmp['ROC_AUC'].iloc[0], df_tmp['PR_AUC'].iloc[0])

        tag = variant.replace('+', '_')
        cnn._save_results_csv(
            df_res, f'minirocket_{tag}_{options.run_name}')


def _parse_script_args():
    """
    Parse the arguments specific to this script and remove them from sys.argv,
    leaving the usual CNN options to the options parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--minirocket-features', type=int, default=2520,
        help='The number of MiniRocket features (rounded down to a multiple '
             'of 84). More features need proportionally more memory: the '
             'dense matrix is (nb events x nb features) in float32.')
    parser.add_argument(
        '--max-negatives', type=int, default=50000,
        help='The number of negative events kept per split (0 = all). All '
             'positives are always kept.')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    return args


def _select_indices(y, max_negatives, rng):
    """
    Select the events to extract: all positives and a random sample of the
    negatives (a dense feature matrix for the full dataset does not fit in
    memory).
    """
    y = np.asarray(y).squeeze()
    idxs_pos = np.where(y > 0)[0]
    idxs_neg = np.where(y == 0)[0]
    if max_negatives and len(idxs_neg) > max_negatives:
        idxs_neg = rng.choice(idxs_neg, size=max_negatives, replace=False)

    # Sorted: the extraction reads the events in dataset order.
    return np.sort(np.concatenate([idxs_pos, idxs_neg]))


def _materialize(dg, idxs):
    """
    Extract the (series, static, target) arrays for the given event indices
    from a data generator.
    """
    all_series, all_static, all_y = [], [], []
    chunks = range(0, len(idxs), BATCH_SIZE_EXTRACT)
    for start in tqdm(chunks, desc="Extracting the precipitation series"):
        x, y = dg._generate_batch(idxs[start:start + BATCH_SIZE_EXTRACT])
        x_static = None
        if isinstance(x, tuple):
            x_3d, x_static = x[0], x[1]
        else:
            x_3d = x
        # (batch, T, 1, 1, 1) -> (batch, T)
        all_series.append(np.nan_to_num(
            x_3d.reshape(x_3d.shape[0], x_3d.shape[1]).astype(np.float32)))
        if x_static is not None:
            all_static.append(np.nan_to_num(x_static.astype(np.float32)))
        all_y.append(np.asarray(y).squeeze(axis=-1))

    series = np.concatenate(all_series)
    static = np.concatenate(all_static) if all_static else None
    y = np.concatenate(all_y).astype(int)

    return series, static, y


def _fit_ridge(x_train, y_train, x_valid, y_valid):
    """
    Fit a ridge classifier, selecting the regularization strength on the
    validation split (ROC-AUC). The lsqr solver is iterative: unlike the
    default cross-validated ridge, it does not decompose the design matrix,
    which is not feasible at this number of samples and features.
    """
    best_clf, best_auc, best_alpha = None, -np.inf, None
    for alpha in ALPHAS:
        clf = RidgeClassifier(alpha=alpha, class_weight='balanced',
                              solver='lsqr')
        clf.fit(x_train, y_train)
        auc = assess_roc_auc(y_valid, clf.decision_function(x_valid))
        logger.info("alpha=%.4g -> validation ROC AUC=%.4f", alpha, auc)
        if auc > best_auc:
            best_clf, best_auc, best_alpha = clf, auc, alpha

    logger.info("Selected alpha=%.4g (validation ROC AUC=%.4f)",
                best_alpha, best_auc)

    return best_clf


def _find_best_f1_threshold(scores, y_obs):
    """
    Select the decision threshold maximizing F1 on the given (validation) set.
    Ridge decision scores are unbounded, so candidates are score quantiles.
    """
    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 201))
    best_thr, best_f1 = 0.0, -np.inf
    eps = 1e-7
    for thr in np.unique(thresholds):
        y_cls = (scores >= thr).astype(int)
        tp = int(np.sum((y_obs == 1) & (y_cls == 1)))
        fp = int(np.sum((y_obs == 0) & (y_cls == 1)))
        fn = int(np.sum((y_obs == 1) & (y_cls == 0)))
        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr


if __name__ == '__main__':
    main()
