"""
Use a CNN model to predict the occurrence of damages to buildings.
"""
from keras import models

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.precip_icon import IconPrecip

config = Config()


def main():
    options = ImpactCnnOptions()
    options.parse_args()
    assert options.is_ok()

    # Load events
    events = None
    # TODO: load events

    # Load the q99 grid for the precipitation data
    q99 = None
    # TODO: load q99 from netcdf file

    # Load precipitation data
    precip = IconPrecip()
    precip.load_data(config.get('DIR_PRECIP_ICON'))
    precip.log_transform()
    precip.normalize(q99)

    # Load the model
    cnn = models.load_model(config.get('MODEL_PATH'))
    cnn.set_precipitation(precip)

    # Predict the occurrence of damages
    predictions = cnn.predict()


if __name__ == '__main__':
    main()
