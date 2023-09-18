"""
Class to compute the impact function.
"""
import numpy as np

from .impact import Impact
from .utils.verification import compute_confusion_matrix, print_classic_scores


class ImpactThresholds(Impact):
    """
    The Impact class using simple thresholds.
    """

    def __init__(self, events):
        super().__init__(events, target_type='occurrence')

        self.thr_i_max = 0.9
        self.thr_p_sum = 0.98
        self.method = 'union'

    def set_thresholds(self, thr_i_max=0.9, thr_p_sum=0.98, method='union'):
        """
        Set the thresholds for the impact function.

        Parameters
        ----------
        thr_i_max: float
            The threshold for the maximum intensity
        thr_p_sum: float
            The threshold for the precipitation sum
        method: str
            The method to use to combine the two thresholds. Options are: 'union',
            'intersection'
        """
        self.thr_i_max = thr_i_max
        self.thr_p_sum = thr_p_sum
        self.method = method

    def _assess_model(self, x, y, period_name):
        """
        Assess the model on a single period.
        """
        # Apply the threshold method
        y_pred = np.zeros(len(y))
        if self.method == 'union':
            y_pred[x[:, 0] >= self.thr_i_max] = 1
            y_pred[x[:, 1] >= self.thr_p_sum] = 1
        elif self.method == 'intersection':
            y_pred[(x[:, 0] >= self.thr_i_max) & (x[:, 1] >= self.thr_p_sum)] = 1

        print(f"\nSplit: {period_name}")

        # Compute the scores
        tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
        print_classic_scores(tp, tn, fp, fn)
        print(f"\n----------------------------------------")
