"""
Tests for the recall-at-a-fixed-alarm-budget score.
"""

import numpy as np
import pytest

from swafi.utils.verification import assess_recall_at_budget


def test_perfect_ranking_catches_everything_at_budget_1x():
    """All positives ranked first: a budget equal to their number catches all."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    scores = assess_recall_at_budget(y_true, y_pred)
    assert scores['recall_at_1x'] == pytest.approx(1.0)
    assert scores['recall_at_2x'] == pytest.approx(1.0)


def test_worst_ranking_catches_nothing_at_budget_1x():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([.0, .1, .2, .9, .8, .7, .6, .5, .4, .3])
    scores = assess_recall_at_budget(y_true, y_pred)
    assert scores['recall_at_1x'] == pytest.approx(0.0)


def test_budget_scales_with_the_number_of_positives():
    """A budget of 2x means twice as many alarms as there are events."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([.9, .85, .8, .75, .7, .6, .5, .4, .3, .2])
    scores = assess_recall_at_budget(y_true, y_pred, budget_multiples=(1, 2))
    # 3 positives -> budget 3 covers ranks 1-3 (2 hits); budget 6 covers 1-6 (3).
    assert scores['recall_at_1x'] == pytest.approx(2 / 3)
    assert scores['recall_at_2x'] == pytest.approx(1.0)


def test_at_budget_1x_recall_equals_precision():
    """
    The budget equals the number of events, so the alarm count equals the
    positive count and the two rates coincide - which is why this value is also
    the F1 score at that operating point.
    """
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=5000) < 0.02).astype(int)
    y_pred = rng.uniform(size=5000) + 0.3 * y_true

    n_pos = int(y_true.sum())
    order = np.argsort(-y_pred, kind='stable')
    tp = int(y_true[order][:n_pos].sum())

    recall = assess_recall_at_budget(y_true, y_pred)['recall_at_1x']
    assert recall == pytest.approx(tp / n_pos)
    assert recall == pytest.approx(tp / n_pos)  # precision, same denominator


def test_no_positives_gives_nan():
    scores = assess_recall_at_budget(np.zeros(10), np.linspace(0, 1, 10))
    assert all(np.isnan(v) for v in scores.values())


def test_budget_is_capped_at_the_sample_size():
    """A 5x budget on a tiny sample cannot exceed the number of cases."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([.9, .1, .8, .2])
    scores = assess_recall_at_budget(y_true, y_pred, budget_multiples=(5,))
    assert scores['recall_at_5x'] == pytest.approx(1.0)
