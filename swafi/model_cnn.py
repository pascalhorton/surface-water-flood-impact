"""
Class to compute the impact function.
"""

import math
from keras import layers, models


class ModelCnn(models.Model):
    """
    CNN model factory.

    Parameters
    ----------
    task: str
        The task. Options are: 'regression', 'classification'
    options: ImpactCnnOptions
        The options.
    input_3d_size: list, None
        The input 3D size.
    input_1d_size: list, None
        The input 1D size.
    """

    def __init__(self, task, options, input_3d_size, input_1d_size):
        super(ModelCnn, self).__init__()
        self.model = None
        self.task = task
        self.options = options

        if input_3d_size is None:
            self.input_3d_size = None
        else:
            self.input_3d_size = list(input_3d_size)

        if input_1d_size is None:
            self.input_1d_size = None
        else:
            self.input_1d_size = list(input_1d_size)

        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        self._check_input_size()
        self._build_model()

    def _check_input_size(self):
        """
        Check the input size.
        """
        if self.input_1d_size is None and self.input_3d_size is None:
            raise ValueError("At least one input size must be provided")

        if self.input_1d_size is not None:
            assert len(self.input_1d_size) == 1, "Input 1D size must be 1D"

        if self.input_3d_size is not None:
            assert len(self.input_3d_size) == 3, \
                "Input 3D size must be 3D (with channels)"

            # Check the input 3D size vs nb_conv_blocks
            input_3d_size = min(self.input_3d_size[0], self.input_3d_size[1])
            nb_conv_blocks_max = math.floor(math.log(input_3d_size, 2))
            if self.options.nb_conv_blocks > nb_conv_blocks_max:
                self.options.nb_conv_blocks = nb_conv_blocks_max
                print(f"Warning: Number of convolution blocks was reduced "
                      f"to {self.options.nb_conv_blocks}")

    def _build_model(self):
        """
        Build the model.
        """
        x = None

        if self.input_3d_size is not None:
            input_3d = layers.Input(shape=self.input_3d_size, name='input_3d')

            # 3D convolution
            x = input_3d
            for i in range(self.options.nb_conv_blocks):
                nb_filters = self.options.nb_filters * (2 ** i)
                kernel_size = (self.options.kernel_size_spatial,
                               self.options.kernel_size_spatial,
                               self.options.kernel_size_temporal)
                pool_size = (self.options.pool_size_spatial,
                             self.options.pool_size_spatial,
                             self.options.pool_size_temporal)
                x = self.conv3d_block(x, i, filters=nb_filters, kernel_size=kernel_size,
                                      pool_size=pool_size)

            # Flatten
            x = layers.Flatten()(x)

        if self.input_1d_size is not None:
            input_1d = layers.Input(shape=self.input_1d_size, name='input_1d')

            if self.input_3d_size is not None:
                # Concatenate with 1D input
                x = layers.concatenate([x, input_1d])
            else:
                x = input_1d

        # Fully connected
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = self.options.nb_dense_units // (2 ** i)
            else:
                nb_units = self.options.nb_dense_units
            x = layers.Dense(nb_units, activation=self.options.inner_activation_dense,
                             name=f'dense_{i}')(x)

            if self.options.with_batchnorm_dense:
                x = layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            if self.options.dropout_rate_dense > 0:
                x = layers.Dropout(rate=self.options.dropout_rate_dense,
                                   name=f'dropout_dense_{i}')(x)

        # Last activation
        output = layers.Dense(1, activation=self.last_activation,
                              name=f'dense_last')(x)

        # Build model
        if self.input_3d_size is not None and self.input_1d_size is not None:
            self.model = models.Model(inputs=[input_3d, input_1d], outputs=output)
        elif self.input_3d_size is None:
            self.model = models.Model(inputs=input_1d, outputs=output)
        elif self.input_1d_size is None:
            self.model = models.Model(inputs=input_3d, outputs=output)
        else:
            raise ValueError("At least one input size must be provided")

    def conv3d_block(self, x, i, filters, kernel_size=(3, 3, 3),
                     initializer='he_normal', activation='default',
                     pool_size=(1, 1, 3)):
        """
        3D convolution block.

        Parameters
        ----------
        x: layers.Layer
            The input layer.
        i: int
            The index of the block.
        filters: int
            The number of filters.
        kernel_size: tuple
            The kernel size (default: (3, 3, 3)).
        initializer: str
            The initializer.
        activation: str
            The activation function.
        pool_size: tuple
            The pool size for the 3D max pooling (default: (1, 1, 3)).

        Returns
        -------
        The output layer.
        """
        if activation == 'default':
            activation = self.options.inner_activation_cnn

        x = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv3d_{i}a',
        )(x)
        x = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv3d_{i}b',
        )(x)

        if self.options.with_batchnorm_cnn:
            # Batch normalization should be before any dropout
            # https://stackoverflow.com/questions/59634780/correct-order-for-
            # spatialdropout2d-batchnormalization-and-activation-function
            x = layers.BatchNormalization(
                name=f'batchnorm_cnn_{i}'
            )(x)

        x = layers.MaxPooling3D(
            pool_size=pool_size,
            name=f'maxpool3d_cnn_{i}',
        )(x)

        if self.options.dropout_rate_cnn > 0:
            if self.options.with_spatial_dropout and x.shape[1] > 1 and x.shape[2] > 1:
                x = layers.SpatialDropout3D(
                    rate=self.options.dropout_rate_cnn,
                    name=f'spatial_dropout_cnn_{i}',
                )(x)
            else:
                x = layers.Dropout(
                    rate=self.options.dropout_rate_cnn,
                    name=f'dropout_cnn_{i}',
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
