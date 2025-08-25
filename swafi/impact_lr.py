"""
Class to compute the impact function.
"""

from .impact import Impact

from sklearn.linear_model import LogisticRegression


class ImpactLogisticRegression(Impact):
    """
    The generic Logistic Regression Impact class.

    Parameters
    ----------
    options: ImpactBasicOptions
        The options.
    events: Events
        The events object.
    """

    def __init__(self, options, events=None):
        super().__init__(options, events)

    def fit(self):
        """
        Train the model.
        """
        self.model = LogisticRegression(class_weight=self.class_weight, max_iter=1000)
        self.model.fit(self.x_train, self.y_train)
