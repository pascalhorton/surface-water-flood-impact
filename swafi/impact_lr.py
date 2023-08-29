"""
Class to compute the impact function.
"""

from .impact import Impact

from sklearn.linear_model import LogisticRegression


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
