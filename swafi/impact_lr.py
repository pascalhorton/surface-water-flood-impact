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
    events: Events
        The events object.
    options: ImpactBasicOptions
        The options.
    """

    def __init__(self, events, options):
        super().__init__(events, options)

    def fit(self):
        """
        Train the model.
        """
        self.model = LogisticRegression(class_weight=self.class_weight, max_iter=1000)
        self.model.fit(self.x_train, self.y_train)
