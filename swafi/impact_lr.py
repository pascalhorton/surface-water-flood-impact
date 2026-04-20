"""
Class to compute the impact function.
"""

from .impact import Impact

import logging
import pickle
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


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

    def save_model(self, dir_output, base_name):
        """
        Save the model.

        Parameters
        ----------
        dir_output: str
            The directory where to save the model.
        base_name: str
            The base name to use for the file.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        filename = f'{dir_output}/{base_name}_{self.options.run_name}.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info("Model saved: %s", filename)

    def load_model(self, dir_output, base_name):
        """
        Load the model.

        Parameters
        ----------
        dir_output: str
            The directory where the model is saved.
        base_name: str
            The base name used for the file.
        """
        filename = f'{dir_output}/{base_name}_{self.options.run_name}.pkl'

        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

        logger.info("Model loaded: %s", filename)
