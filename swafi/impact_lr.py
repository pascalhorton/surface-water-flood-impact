"""
Class to compute the impact function.
"""

from .impact import Impact

import hashlib
import pickle
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from .utils.plotting import plot_random_forest_feature_importance
from .utils.verification import compute_confusion_matrix, compute_score_binary


class ImpactLogisticRegression(Impact):
    """
    The generic Logistic Regression Impact class.
    """

    def __init__(self, events):
        super().__init__(events)

    def fit(self):
        """
        Train the model.
        """
        self.model = LogisticRegression(class_weight=self.class_weight)
        self.model.fit(self.x_train, self.y_train)
