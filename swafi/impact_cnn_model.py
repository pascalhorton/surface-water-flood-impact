"""
Class for the CNN model.
"""

import math
import keras


class ModelCnn(keras.models.Model):
    """
    CNN model factory.

    Parameters
    ----------
    trainable: bool
        Whether the model is trainable.
    dtype: str|None
        The data type.
    task: str
        The task. Options are: 'regression', 'classification' (default: 'classification').
    options: ImpactCnnOptions|None
        The options.
    input_3d_size: list|None
        The input 3D size (default: None).
    input_1d_size: list|None
        The input 1D size (default: None).
    *args
    **kwargs
        Additional arguments to pass to keras.models.Model.
    """

    def __init__(self, trainable=True, dtype=None, task='classification', options=None, input_3d_size=None, input_1d_size=None, *args, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, *args, **kwargs)
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

    def get_config(self):
        """
        Return a serializable config for this wrapper.
        Ensure `self.options` is serializable (implements get_config) or is primitive.
        """
        base_config = super().get_config()
        config = {
            "task": self.task,
            "options": keras.saving.serialize_keras_object(self.options),
            "input_3d_size": self.input_3d_size,
            "input_1d_size": self.input_1d_size,
            "build_config": self.get_build_config(),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Recreate the wrapper and build the internal Keras model.
        Keras will call this when deserializing the custom object.
        """
        # Extract config values (keep backward compatibility)
        trainable = config.get("trainable", True)
        dtype = config.get("dtype", None)

        options_cfg = config.get("options", None)
        options = None
        if options_cfg is not None:
            # restore options if it was serialized
            options = keras.saving.deserialize_keras_object(options_cfg)

        # Recreate instance
        instance = cls(trainable=trainable, dtype=dtype)
        instance.task = config.get("task", "classification")
        instance.options = options
        instance.input_3d_size = config.get("input_3d_size", None)
        instance.input_1d_size = config.get("input_1d_size", None)
        instance.last_activation = 'relu' if instance.task == 'regression' else 'sigmoid'

        return instance

    def get_build_config(self):
        """
        Return the internal built model config so Keras can persist it.
        """
        if self.model is None:
            return None
        return self.model.get_config()

    def build_from_config(self, config):
        """
        Complementary to get_build_config. Rebuilds wrapper + internal model.
        """
        input_shape = config.get("input_shape", None)
        self.input_3d_size = input_shape[0][1:] if input_shape and len(input_shape) > 0 else None
        self.input_1d_size = input_shape[1][1:] if input_shape and len(input_shape) > 1 else None

        # Try to restore the nested keras model from a stored build config
        build_cfg = config.get("build_config", None)
        if build_cfg is not None:
            try:
                self.model = keras.models.Model.from_config(build_cfg)
            except Exception:
                # if reconstruction fails, leave model None and fall back to build_model below
                self.model = None

    def build_model(self, options=None):
        """
        Build the model.

        Parameters
        ----------
        options: ImpactCnnOptions
            The options.
        """
        if options is not None:
            self.options = options

        self._check_input_size()

        x = None

        if self.input_3d_size is not None:
            input_3d = keras.layers.Input(shape=self.input_3d_size, name='input_3d')

            if not self.options.use_3d_cnn:
                # If 3D CNN is not used, remove the last dimension (channels)
                x = keras.layers.Reshape(
                    (self.input_3d_size[0], self.input_3d_size[1], self.input_3d_size[2]),
                    name='reshape_input_3d'
                )(input_3d)
            else:
                x = input_3d

            # Convolution
            for i in range(self.options.nb_conv_blocks):
                nb_filters = self.options.nb_filters * (2 ** i)
                if self.options.use_3d_cnn:
                    kernel_size = (self.options.kernel_size_spatial,
                                   self.options.kernel_size_spatial,
                                   self.options.kernel_size_temporal)
                    pool_size = (self.options.pool_size_spatial,
                                 self.options.pool_size_spatial,
                                 self.options.pool_size_temporal)
                    x = self._conv3d_block(
                        x, i,
                        filters=nb_filters,
                        kernel_size=kernel_size,
                        pool_size=pool_size
                    )
                else:
                    x = self._conv2d_block(
                        x, i,
                        filters=nb_filters,
                        kernel_size=self.options.kernel_size_spatial,
                        pool_size=self.options.pool_size_spatial
                    )

            # Flatten
            x = keras.layers.Flatten()(x)

        if self.input_1d_size is not None:
            input_1d = keras.layers.Input(shape=self.input_1d_size, name='input_1d')
            if self.input_3d_size is not None:
                x = keras.layers.concatenate([x, input_1d])
            else:
                x = input_1d

        # Fully connected
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = self.options.nb_dense_units // (2 ** i)
                # Keep at least 4 units
                nb_units = max(nb_units, 4)
            else:
                nb_units = self.options.nb_dense_units
            x = keras.layers.Dense(nb_units, activation=self.options.inner_activation_dense,
                             name=f'dense_{i}')(x)

            if self.options.use_batchnorm_dense:
                x = keras.layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            if self.options.dropout_rate_dense > 0:
                x = keras.layers.Dropout(rate=self.options.dropout_rate_dense,
                                   name=f'dropout_dense_{i}')(x)

        # Last activation
        output = keras.layers.Dense(1, activation=self.last_activation,
                              name=f'dense_last')(x)

        # Build model
        if self.input_3d_size is not None and self.input_1d_size is not None:
            self.model = keras.models.Model(inputs=[input_3d, input_1d], outputs=output)
        elif self.input_3d_size is None:
            self.model = keras.models.Model(inputs=input_1d, outputs=output)
        elif self.input_1d_size is None:
            self.model = keras.models.Model(inputs=input_3d, outputs=output)
        else:
            raise ValueError("At least one input size must be provided")

    def _setup(self, task='classification', options=None, input_3d_size=None, input_1d_size=None):
        """
        Setup the model.

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


        self._check_input_size()

    def _check_input_size(self):
        """
        Check the input size.
        """
        if self.input_1d_size is None and self.input_3d_size is None:
            raise ValueError("At least one input size must be provided")

        if self.input_1d_size is not None:
            assert len(self.input_1d_size) == 1, "Input 1D size must be 1D"

        if self.input_3d_size is not None:
            assert len(self.input_3d_size) == 4, \
                "Input 3D size must be 4D (with channels)"

            if self.options is None:
                return

            # Check the input 3D size vs nb_conv_blocks
            nb_conv_blocks_max = self.options.nb_conv_blocks
            if self.options.pool_size_spatial > 1:
                spatial_size = min(self.input_3d_size[0], self.input_3d_size[1])
                nb_conv_blocks_max = min(
                    nb_conv_blocks_max, math.floor(
                        math.log(spatial_size, self.options.pool_size_spatial)))
            if self.options.pool_size_temporal > 1:
                nb_conv_blocks_max = min(
                    nb_conv_blocks_max, math.floor(
                        math.log(self.input_3d_size[2],
                                 self.options.pool_size_temporal)))
            if self.options.nb_conv_blocks > nb_conv_blocks_max:
                self.options.nb_conv_blocks = nb_conv_blocks_max
                print(f"Warning: Number of convolution blocks was reduced "
                      f"to {self.options.nb_conv_blocks}")

    def _conv3d_block(self, x, i, filters, kernel_size=(3, 3, 3),
                      initializer='he_normal', activation='default',
                      pool_size=(1, 1, 3)):
        """
        3D convolution block.

        Parameters
        ----------
        x: keras.layers.Layer
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

        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv3d_{i}a',
        )(x)
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv3d_{i}b',
        )(x)

        if self.options.use_batchnorm_cnn:
            # Batch normalization should be before any dropout
            # https://stackoverflow.com/questions/59634780/correct-order-for-
            # spatialdropout2d-batchnormalization-and-activation-function
            x = keras.layers.BatchNormalization(
                name=f'batchnorm_cnn_{i}'
            )(x)

        x = keras.layers.MaxPooling3D(
            pool_size=pool_size,
            name=f'maxpool3d_cnn_{i}',
        )(x)

        if self.options.dropout_rate_cnn > 0:
            if self.options.use_spatial_dropout and x.shape[1] > 1 and x.shape[2] > 1:
                x = keras.layers.SpatialDropout3D(
                    rate=self.options.dropout_rate_cnn,
                    name=f'spatial_dropout_cnn_{i}',
                )(x)
            else:
                x = keras.layers.Dropout(
                    rate=self.options.dropout_rate_cnn,
                    name=f'dropout_cnn_{i}',
                )(x)

        return x

    def _conv2d_block(self, x, i, filters, kernel_size=3,
                      initializer='he_normal', activation='default',
                      pool_size=2):
        """
        2D convolution block.

        Parameters
        ----------
        x: keras.layers.Layer
            The input layer.
        i: int
            The index of the block.
        filters: int
            The number of filters.
        kernel_size: int
            The kernel size (default: 3).
        initializer: str
            The initializer.
        activation: str
            The activation function.
        pool_size: int
            The pool size for the 2D max pooling (default: 2).

        Returns
        -------
        The output layer.
        """
        if activation == 'default':
            activation = self.options.inner_activation_cnn

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv2d_{i}a',
        )(x)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
            name=f'conv2d_{i}b',
        )(x)

        if self.options.use_batchnorm_cnn:
            x = keras.layers.BatchNormalization(
                name=f'batchnorm_cnn_{i}'
            )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=pool_size,
            name=f'maxpool2d_cnn_{i}',
        )(x)

        if self.options.dropout_rate_cnn > 0:
            if self.options.use_spatial_dropout and x.shape[1] > 1 and x.shape[2] > 1:
                x = keras.layers.SpatialDropout2D(
                    rate=self.options.dropout_rate_cnn,
                    name=f'spatial_dropout_cnn_{i}',
                )(x)
            else:
                x = keras.layers.Dropout(
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
