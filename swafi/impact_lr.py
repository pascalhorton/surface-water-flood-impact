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
    random_state: int|None
        The random state to use for the random number generator.
        Default: None. Set to None to not set the random seed.
    """

    def __init__(self, events, random_state=None):
        super().__init__(events, target_type='occurrence', random_state=random_state)

    def fit(self):
        """
        Train the model.
        """
        self.model = LogisticRegression(class_weight=self.class_weight)
        self.model.fit(self.x_train, self.y_train)
