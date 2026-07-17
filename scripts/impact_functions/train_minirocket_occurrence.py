"""
Train a MiniRocket + ridge classifier on the at-cell precipitation series to
predict the occurrence of damages to buildings.

Diagnostic companion to the CNN model: MiniRocket features are computed from
the same single-pixel precipitation series the CNN sees, and a linear ridge
classifier is trained on (a) the MiniRocket features, (b) the tabular
features, and (c) both. Comparing the three answers whether the raw series
carries signal beyond the tabular event summaries.
"""

import logging
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

NUM_FEATURES = 9996
BATCH_SIZE_EXTRACT = 1024

config = Config()


def main():
    setup_logging(script_name='train_minirocket_occurrence')
    options = ImpactCnnOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.use_precip, "MiniRocket needs the precipitation series."
    assert options.precip_window_size == options.precip_resolution, \
        "MiniRocket is univariate; use a single-pixel window."
    assert not options.use_poisson_head, \
        "The Poisson head does not apply to the ridge classifier."

    if options.random_state is not None:
        random.seed(options.random_state)
        np.random.seed(options.random_state)

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

    logger.info("Extracting precipitation series per split.")
    series, static, targets = {}, {}, {}
    for name, dg in splits.items():
        series[name], static[name], targets[name] = _materialize(dg)
        logger.info("%s: %d events, series length %d",
                    name, len(targets[name]), series[name].shape[1])

    logger.info("Fitting MiniRocket on the training series.")
    rocket = MiniRocket(num_features=NUM_FEATURES,
                        random_state=options.random_state)
    rocket.fit(series['train'])
    rocket_feats = {name: rocket.transform(s) for name, s in series.items()}

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
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                              class_weight='balanced'))
        clf.fit(feats['train'], targets['train'])

        scores = {name: clf.decision_function(f) for name, f in feats.items()}
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
            df_res = pd.concat([df_res, df_tmp])

        tag = variant.replace('+', '_')
        cnn._save_results_csv(
            df_res, f'minirocket_{tag}_{options.run_name}')


def _materialize(dg):
    """
    Extract the full (series, static, target) arrays from a data generator,
    in dataset order.
    """
    n = len(dg.y)
    all_series, all_static, all_y = [], [], []
    for start in range(0, n, BATCH_SIZE_EXTRACT):
        idxs = np.arange(start, min(start + BATCH_SIZE_EXTRACT, n))
        x, y = dg._generate_batch(idxs)
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
