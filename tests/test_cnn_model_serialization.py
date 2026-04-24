import numpy as np
import pytest
import keras

from swafi.impact_cnn_model import ModelCnn
from swafi.impact_cnn_options import ImpactCnnOptions


def make_options(pixels_per_side: int = 1) -> ImpactCnnOptions:
    """Create a minimal ImpactCnnOptions instance with required attributes set.
    We intentionally avoid calling parse_args() to prevent pytest CLI arg conflicts.
    Only attributes accessed by ModelCnn.build_model / _check_input_size are set.
    """
    opts = ImpactCnnOptions()

    # Convolutional / spatial CNN options
    opts.nb_conv_blocks = 1
    opts.nb_filters = 8
    opts.kernel_size_spatial = 3
    opts.pool_size_spatial = 1
    opts.dropout_rate_cnn = 0.0
    opts.use_spatial_dropout = False
    opts.use_batchnorm_cnn = False
    opts.inner_activation_cnn = 'relu'

    # TCN options
    opts.tcn_filters = 16
    opts.tcn_kernel_size = 3
    opts.tcn_nb_layers = 2
    opts.dropout_rate_tcn = 0.0

    # Dense layers
    opts.nb_dense_layers = 1
    opts.nb_dense_units = 16
    opts.nb_dense_units_decreasing = False
    opts.inner_activation_dense = 'relu'
    opts.dropout_rate_dense = 0.0
    opts.use_batchnorm_dense = False

    # Misc flags
    opts.use_precip = True
    opts.precip_window_size = pixels_per_side
    opts.precip_resolution = 1
    opts.precip_time_step = 1
    opts.precip_days_before = 1
    opts.precip_days_after = 0

    return opts


@pytest.mark.parametrize("pixels_per_side", [1, 4])
def test_model_cnn_serialization_roundtrip(tmp_path, pixels_per_side):
    np.random.seed(0)

    options = make_options(pixels_per_side)

    input_3d_size = [4, pixels_per_side, pixels_per_side, 1]
    input_1d_size = [10]

    model = ModelCnn(task='classification', options=options,
                     input_3d_size=input_3d_size, input_1d_size=input_1d_size)
    model.build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Dummy data (batch size 2)
    x3d = np.random.rand(2, *input_3d_size).astype('float32')
    x1d = np.random.rand(2, *input_1d_size).astype('float32')
    y = np.array([0., 1.], dtype='float32')

    # One quick training step
    model.fit([x3d, x1d], y, epochs=1, verbose=0)

    # Save without optimizer state to avoid warnings (warnings are errors via pytest config)
    save_path = tmp_path / f'model_px{pixels_per_side}.keras'
    model.save(save_path, include_optimizer=False)

    # Load and predict
    loaded = keras.models.load_model(save_path)
    preds = loaded.predict([x3d, x1d], verbose=0)

    assert preds.shape == (2, 1), f"Unexpected prediction shape: {preds.shape}"
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "Predictions not in [0,1] for sigmoid output"

