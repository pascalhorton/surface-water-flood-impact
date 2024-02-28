"""
Class to compute the impact function.
"""

import math
import warnings
from keras import layers, models


class DeepImpact(models.Model):
    """
    Model factory.

    Parameters
    ----------
    task: str
        The task. Options are: 'regression', 'classification'
    input_2d_size: ?list
        The input 2D size.
    input_1d_size: ?list
        The input 1D size.
    dropout_rate: float
        The dropout rate.
    with_spatial_dropout: bool
        Whether to use spatial dropout or not.
    with_batchnorm: bool
        Whether to use batch normalization or not.
    inner_activation: str
        The inner activation function.
    """

    def __init__(self, task, input_2d_size, input_1d_size, dropout_rate=0.2,
                 with_spatial_dropout=True, with_batchnorm=True,
                 inner_activation='relu'):
        super(DeepImpact, self).__init__()
        self.model = None
        self.task = task
        if input_2d_size is None:
            self.input_2d_size = None
        else:
            self.input_2d_size = list(input_2d_size)
        if input_1d_size is None:
            self.input_1d_size = None
        else:
            self.input_1d_size = list(input_1d_size)
        self.dropout_rate = dropout_rate
        self.nb_conv_blocks = 3
        self.nb_filters = 64
        self.with_spatial_dropout = with_spatial_dropout
        self.with_batchnorm = with_batchnorm
        self.inner_activation = inner_activation
        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        self._check_input_size()
        self._build_model()

    def _check_input_size(self):
        """
        Check the input size.
        """
        if self.input_1d_size is None and self.input_2d_size is None:
            raise ValueError("At least one input size must be provided")

        if self.input_1d_size is not None:
            assert len(self.input_1d_size) == 1, "Input 1D size must be 1D"

        if self.input_2d_size is not None:
            assert len(self.input_2d_size) == 3, \
                "Input 2D size must be 3D (with channels)"

            # Check the input 2D size vs nb_conv_blocks
            input_2d_size = min(self.input_2d_size[0], self.input_2d_size[1])
            nb_conv_blocks_max = math.floor(math.log(input_2d_size, 2))
            if self.nb_conv_blocks > nb_conv_blocks_max:
                self.nb_conv_blocks = nb_conv_blocks_max
                warnings.warn(f"Number of convolution blocks was reduced "
                              f"to {self.nb_conv_blocks}")

    def _build_model(self):
        """
        Build the model.
        """
        x = None

        if self.input_2d_size is not None:
            input_2d = layers.Input(shape=self.input_2d_size, name='input_2d')

            # 2D convolution
            x = input_2d
            for i in range(self.nb_conv_blocks):
                nb_filters = self.nb_filters * (2 ** i)
                x = self.conv2d_block(x, filters=nb_filters, kernel_size=3)

            # Flatten
            x = layers.Flatten()(x)

        if self.input_1d_size is not None:
            input_1d = layers.Input(shape=self.input_1d_size, name='input_1d')

            if self.input_2d_size is not None:
                # Concatenate with 1D input
                x = layers.concatenate([x, input_1d])
            else:
                x = input_1d

        # Fully connected
        x = layers.Dense(256, activation=self.inner_activation)(x)
        x = layers.Dense(128, activation=self.inner_activation)(x)
        x = layers.Dense(64, activation=self.inner_activation)(x)
        x = layers.Dense(32, activation=self.inner_activation)(x)

        # Last activation
        output = layers.Dense(1, activation=self.last_activation)(x)

        # Build model
        if self.input_2d_size is not None and self.input_1d_size is not None:
            self.model = models.Model(inputs=[input_2d, input_1d], outputs=output)
        elif self.input_2d_size is None:
            self.model = models.Model(inputs=input_1d, outputs=output)
        elif self.input_1d_size is None:
            self.model = models.Model(inputs=input_2d, outputs=output)
        else:
            raise ValueError("At least one input size must be provided")

    def conv2d_block(self, x, filters, kernel_size=3, initializer='he_normal',
                     activation='default'):
        """
        Convolution block.

        Parameters
        ----------
        x: layers.Layer
            The input layer.
        filters: int
            The number of filters.
        kernel_size: int
            The kernel size.
        initializer: str
            The initializer.
        activation: str
            The activation function.

        Returns
        -------
        The output layer.
        """
        if activation == 'default':
            activation = self.inner_activation

        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
        )(x)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation=activation,
        )(x)

        if self.with_batchnorm:
            # Batch normalization should be before any dropout
            # https://stackoverflow.com/questions/59634780/correct-order-for-
            # spatialdropout2d-batchnormalization-and-activation-function
            x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(
            pool_size=(2, 2),
        )(x)

        if self.dropout_rate > 0:
            if self.with_spatial_dropout and x.shape[1] > 1 and x.shape[2] > 1:
                x = layers.SpatialDropout2D(
                    rate=self.dropout_rate,
                )(x)
            else:
                x = layers.Dropout(
                    rate=self.dropout_rate,
                )(x)

        return x

    def call(self, inputs, **kwargs):
        """
        Call the model.

        Parameters
        ----------
        inputs: list
            The inputs.
        **kwargs

        Returns
        -------
        The output.
        """
        return self.model(inputs, **kwargs)
