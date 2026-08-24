"""
Tests for the flag that marks rows whose threshold-derived scores describe the
threshold search rather than the model.
"""

import numpy as np

from swafi.impact import Impact

flag = Impact._flag_degenerate_predictions


def _spread_preds(n=1000, seed=0):
    """Predictions with a real spread, i.e. a model that ranks."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


def test_healthy_run_is_not_flagged():
    """A normal operating point: some TP, some FP, some TN, some FN."""
    assert not flag(_spread_preds(), roc=0.914, tp=208, tn=279845, fp=1177,
                    fn=826, period_name='valid')


def test_no_ranking_is_flagged():
    """Near-constant output, ROC-AUC at chance."""
    assert flag(np.full(1000, 0.5), roc=0.501, tp=1034, tn=0, fp=278426, fn=0,
                period_name='valid')


def test_threshold_above_every_prediction_is_flagged():
    """
    The case that keying on ROC-AUC alone misses: the model ranks well
    (ROC-AUC 0.90) but the tuned threshold sits above every predicted
    probability, so nothing is called positive. Numbers are from the GVZ hourly
    logistic regression on the event precipitation summaries at
    weight_denominator 20, which reported degenerate=False before this check
    existed.
    """
    assert flag(_spread_preds(), roc=0.9016, tp=0, tn=278426, fp=0, fn=1034,
                period_name='valid')


def test_threshold_below_every_prediction_is_flagged():
    """The mirror image: everything is labelled positive."""
    assert flag(_spread_preds(), roc=0.9016, tp=1034, tn=0, fp=278426, fn=0,
                period_name='valid')


def test_non_finite_roc_is_flagged():
    assert flag(_spread_preds(), roc=float('nan'), tp=208, tn=279845, fp=1177,
                fn=826, period_name='valid')
