"""
Class for the CNN model.
"""

import logging

import keras
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="swafi")
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

    def __init__(self, trainable=True, dtype=None, task='classification', options=None, input_3d_size=None, input_1d_size=None, input_1d_splits=None, output_bias_init=0.0, *args, **kwargs):
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

        self.input_1d_splits = list(input_1d_splits) if input_1d_splits is not None else None
        self.output_bias_init = float(output_bias_init)

        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        # Training-set feature statistics for inference-time consistency.
        self.mean_static = None
        self.std_static = None
        self.min_static = None
        self.max_static = None
        self.mean_precip = None
        self.std_precip = None
        self.q99_precip = None

    def set_feature_stats(self, mean_static=None, std_static=None, min_static=None, max_static=None,
                          mean_precip=None, std_precip=None, q99_precip=None):
        self.mean_static = mean_static
        self.std_static = std_static
        self.min_static = min_static
        self.max_static = max_static
        self.mean_precip = mean_precip
        self.std_precip = std_precip
        self.q99_precip = q99_precip

    @staticmethod
    def _serialize_array(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _deserialize_array(value):
        if value is None:
            return None
        if isinstance(value, list):
            return np.asarray(value)
        return value

    def get_config(self):
        """
        Return a serializable config for this wrapper.
        Ensure `self.options` is serializable (implements get_config) or is primitive.
        """
        base_config = super().get_config()
        try:
            options_cfg = (keras.saving.serialize_keras_object(self.options)
                           if self.options is not None else None)
        except Exception:
            # Fallback: attempt to use a shallow attribute snapshot
            options_cfg = getattr(self.options, 'get_config', lambda: None)()

        config = {
            "task": self.task,
            "options": options_cfg,
            "input_3d_size": self.input_3d_size,
            "input_1d_size": self.input_1d_size,
            "input_1d_splits": self.input_1d_splits,
            "output_bias_init": self.output_bias_init,
            "build_config": self.get_build_config(),
            "mean_static": self._serialize_array(self.mean_static),
            "std_static": self._serialize_array(self.std_static),
            "min_static": self._serialize_array(self.min_static),
            "max_static": self._serialize_array(self.max_static),
            "mean_precip": self._serialize_array(self.mean_precip),
            "std_precip": self._serialize_array(self.std_precip),
            "q99_precip": self._serialize_array(self.q99_precip),
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
            try:
                options = keras.saving.deserialize_keras_object(options_cfg)
            except Exception:
                options = None

        # Recreate instance
        instance = cls(trainable=trainable, dtype=dtype)
        instance.task = config.get("task", "classification")
        instance.options = options
        instance.input_3d_size = config.get("input_3d_size", None)
        instance.input_1d_size = config.get("input_1d_size", None)
        instance.input_1d_splits = config.get("input_1d_splits", None)
        instance.last_activation = 'relu' if instance.task == 'regression' else 'sigmoid'
        instance.output_bias_init = config.get("output_bias_init", 0.0)

        instance.mean_static = cls._deserialize_array(config.get("mean_static", None))
        instance.std_static = cls._deserialize_array(config.get("std_static", None))
        instance.min_static = cls._deserialize_array(config.get("min_static", None))
        instance.max_static = cls._deserialize_array(config.get("max_static", None))
        instance.mean_precip = cls._deserialize_array(config.get("mean_precip", None))
        instance.std_precip = cls._deserialize_array(config.get("std_precip", None))
        instance.q99_precip = cls._deserialize_array(config.get("q99_precip", None))

        # Attempt to rebuild internal functional model from stored config
        build_cfg = config.get("build_config", None)
        if build_cfg is not None:
            try:
                instance.model = keras.models.Model.from_config(build_cfg)
            except Exception:
                instance.model = None  # Will need manual rebuild if used

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
            t_len = self.input_3d_size[0]
            pixels_per_side = self.input_3d_size[1]

            n_channels = self.input_3d_size[3]

            if pixels_per_side > 1:
                # Spatial 2D CNN applied per time step via TimeDistributed
                # Input is already (T, H, W, C) — no permutation needed.
                x = input_3d
                for i in range(self.options.nb_conv_blocks):
                    nb_filters = self.options.nb_filters * (2 ** i)
                    # Conv2D (no activation) → optional BN → Activation → optional pool → optional dropout
                    x = keras.layers.TimeDistributed(
                        keras.layers.Conv2D(
                            nb_filters,
                            self.options.kernel_size_spatial,
                            padding='same',
                            kernel_initializer='he_normal'
                        ),
                        name=f'td_conv2d_{i}'
                    )(x)
                    if self.options.use_batchnorm_cnn:
                        x = keras.layers.TimeDistributed(
                            keras.layers.BatchNormalization(),
                            name=f'td_bn_{i}'
                        )(x)
                    x = keras.layers.TimeDistributed(
                        keras.layers.Activation(self.options.inner_activation_cnn),
                        name=f'td_act_{i}'
                    )(x)
                    if self.options.pool_size_spatial > 1:
                        x = keras.layers.TimeDistributed(
                            keras.layers.MaxPooling2D(pool_size=self.options.pool_size_spatial),
                            name=f'td_pool_{i}'
                        )(x)
                    if self.options.dropout_rate_cnn > 0:
                        if self.options.use_spatial_dropout:
                            x = keras.layers.TimeDistributed(
                                keras.layers.SpatialDropout2D(rate=self.options.dropout_rate_cnn),
                                name=f'td_drop_{i}'
                            )(x)
                        else:
                            x = keras.layers.TimeDistributed(
                                keras.layers.Dropout(rate=self.options.dropout_rate_cnn),
                                name=f'td_drop_{i}'
                            )(x)
                # Flatten spatial dims per time step → (T, spatial_features)
                x = keras.layers.TimeDistributed(
                    keras.layers.Flatten(), name='td_flatten'
                )(x)
            else:
                # 1×1 spatial: squeeze to (T, C)
                x = keras.layers.Reshape((t_len, n_channels), name='reshape_1px')(input_3d)

            # Project to TCN input dimension → (T, tcn_filters)
            x = keras.layers.Dense(self.options.tcn_filters, name='tcn_proj')(x)

            # TCN blocks with exponentially increasing dilation rates
            for i in range(self.options.tcn_nb_layers):
                x = self._tcn_block(x, dilation_rate=2 ** i,
                                    filters=self.options.tcn_filters,
                                    kernel_size=self.options.tcn_kernel_size, i=i)

            # Temporal pooling
            pooling = getattr(self.options, 'tcn_pooling', 'max')
            if pooling == 'mean_max':
                # Max alone keeps peak intensity and discards duration; the mean
                # carries accumulation. Both matter for surface water flooding,
                # so concatenate them rather than choosing.
                x = keras.layers.Concatenate(name='temporal_mean_max')([
                    keras.layers.GlobalAveragePooling1D(name='temporal_mean')(x),
                    keras.layers.GlobalMaxPooling1D(name='temporal_max')(x),
                ])
            elif pooling == 'mean':
                x = keras.layers.GlobalAveragePooling1D(name='temporal_mean')(x)
            elif pooling == 'last':
                x = keras.layers.Lambda(
                    lambda t: t[:, -1, :], name='temporal_last')(x)
            elif pooling == 'attention':
                weights = keras.layers.Dense(1, name='attn_w')(x)
                weights = keras.layers.Softmax(axis=1, name='attn_softmax')(weights)
                x = keras.layers.Multiply(name='attn_mul')([x, weights])
                x = keras.layers.Lambda(
                    lambda t: tf.reduce_sum(t, axis=1), name='attn_sum')(x)
            else:  # 'max' (default)
                x = keras.layers.GlobalMaxPooling1D(name='temporal_max')(x)

        if self.input_1d_size is not None:
            input_1d = keras.layers.Input(shape=self.input_1d_size, name='input_1d')

            use_emb = (
                getattr(self.options, 'use_feature_class_embedding', False)
                and self.input_1d_splits is not None
                and len(self.input_1d_splits) > 1
                and self.input_3d_size is None  # embedding only in pure ANN mode
            )

            if use_emb:
                emb_size = getattr(self.options, 'feature_class_embedding_size', 32)
                sub_tensors = FeatureSplit(
                    split_sizes=self.input_1d_splits,
                    axis=-1,
                    name='feature_split'
                )(input_1d)
                emb_act = getattr(self.options, 'inner_activation_dense', 'relu')
                embeddings = [
                    keras.layers.Dense(emb_size, activation=emb_act,
                                       name=f'emb_{i}')(sub)
                    for i, sub in enumerate(sub_tensors)
                ]
                x1d = keras.layers.Concatenate(name='emb_concat')(embeddings)
            else:
                x1d = input_1d

            if self.input_3d_size is not None:
                x = keras.layers.concatenate([x, x1d])
            else:
                x = x1d

        # Fully connected
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = self.options.nb_dense_units // (2 ** i)
                nb_units = max(nb_units, 4)
            else:
                nb_units = self.options.nb_dense_units

            x_skip = x  # save for residual

            x = keras.layers.Dense(nb_units, name=f'dense_{i}')(x)

            if getattr(self.options, 'use_layernorm_dense', False):
                x = keras.layers.LayerNormalization(name=f'layernorm_dense_{i}')(x)
            elif self.options.use_batchnorm_dense:
                x = keras.layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            x = keras.layers.Activation(
                self.options.inner_activation_dense, name=f'act_dense_{i}'
            )(x)

            if self.options.dropout_rate_dense > 0:
                x = keras.layers.Dropout(rate=self.options.dropout_rate_dense,
                                   name=f'dropout_dense_{i}')(x)

            if getattr(self.options, 'use_residual_dense', False):
                if x_skip.shape[-1] == nb_units:
                    x = keras.layers.Add(name=f'res_dense_{i}')([x, x_skip])
                else:
                    x_proj = keras.layers.Dense(
                        nb_units, use_bias=False, name=f'res_proj_{i}'
                    )(x_skip)
                    x = keras.layers.Add(name=f'res_dense_{i}')([x, x_proj])

        inputs = []
        if self.input_3d_size is not None:
            inputs.append(input_3d)
        if self.input_1d_size is not None:
            inputs.append(input_1d)
        if not inputs:
            raise ValueError("At least one input size must be provided")

        if getattr(self.options, 'use_poisson_head', False):
            # Poisson head: lambda = exp(log_rate + log(nb_contracts)), so the
            # exposure enters as an additive offset in log space.
            log_rate = keras.layers.Dense(
                1,
                activation='linear',
                bias_initializer=keras.initializers.Constant(self.output_bias_init),
                name='dense_last'
            )(x)
            input_offset = keras.layers.Input(shape=(1,), name='input_offset')
            inputs.append(input_offset)
            output = keras.layers.Add(name='add_offset')([log_rate, input_offset])
            output = keras.layers.Activation('exponential', name='lambda')(output)
        else:
            # Last activation — bias initialized to log-odds of class prior for faster convergence
            output = keras.layers.Dense(
                1,
                activation=self.last_activation,
                bias_initializer=keras.initializers.Constant(self.output_bias_init),
                name='dense_last'
            )(x)

        # Build model
        self.model = keras.models.Model(
            inputs=inputs if len(inputs) > 1 else inputs[0], outputs=output)

    def call(self, inputs, training=None, **kwargs):
        """
        Call the model.

        Parameters
        ----------
        inputs: list
            The inputs.
        training: bool
            Whether the model is in training mode.

        Returns
        -------
        The output.
        """
        if self.model is None:
            raise ValueError("Model not defined")
        return self.model(inputs, training=training, **kwargs)

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

            # Guard against invalid dimensions.
            if any(dim is None or dim <= 0 for dim in self.input_3d_size):
                raise ValueError(
                    f"Input 3D size dimensions must be > 0, got {self.input_3d_size}"
                )

            if self.options is None:
                return

            # Cap nb_conv_blocks so no intermediate pooled spatial dim is
            # non-divisible by pool_size. Odd/non-aligned dims (e.g. 10÷2=5)
            # cause cuDNN's fused conv kernel to fail on RTX 4090 / cuDNN 9.5.
            if self.options.pool_size_spatial > 1:
                spatial_size = min(self.input_3d_size[1], self.input_3d_size[2])
                nb_conv_blocks_max = 0
                s = spatial_size
                while s % self.options.pool_size_spatial == 0:
                    nb_conv_blocks_max += 1
                    s //= self.options.pool_size_spatial
                nb_conv_blocks_max = max(1, nb_conv_blocks_max)
                if self.options.nb_conv_blocks > nb_conv_blocks_max:
                    self.options.nb_conv_blocks = nb_conv_blocks_max
                    logger.warning(
                        "Number of convolution blocks was reduced to %s "
                        "(spatial %s must be divisible by pool_size^nb_conv_blocks)",
                        self.options.nb_conv_blocks, spatial_size)

    def _tcn_block(self, x, dilation_rate, filters, kernel_size, i):
        """
        Temporal Convolutional Network (TCN) block with dilated causal Conv1D
        and a residual connection.

        Parameters
        ----------
        x: tensor
            Input tensor of shape (batch, T, features).
        dilation_rate: int
            Dilation rate for the Conv1D.
        filters: int
            Number of Conv1D filters.
        kernel_size: int
            Kernel size for the Conv1D.
        i: int
            Block index (used for layer naming).

        Returns
        -------
        Output tensor of shape (batch, T, filters).
        """
        residual = x
        for j in range(2):
            x = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f'tcn_conv_{i}_{j}'
            )(x)
            x = keras.layers.LayerNormalization(name=f'tcn_ln_{i}_{j}')(x)
            x = keras.layers.Activation(
                self.options.inner_activation_cnn, name=f'tcn_act_{i}_{j}'
            )(x)
            if self.options.dropout_rate_tcn > 0:
                x = keras.layers.Dropout(
                    rate=self.options.dropout_rate_tcn, name=f'tcn_drop_{i}_{j}'
                )(x)
        # Residual: 1×1 conv to match dimensions if needed
        if residual.shape[-1] != filters:
            residual = keras.layers.Conv1D(
                filters, 1, kernel_initializer='he_normal', name=f'tcn_res_{i}'
            )(residual)
        return keras.layers.Add(name=f'tcn_add_{i}')([x, residual])


@keras.saving.register_keras_serializable(package="swafi")
class FeatureSplit(keras.layers.Layer):
    """Serializable wrapper around tf.split for feature-class embedding."""

    def __init__(self, split_sizes, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.split_sizes = [int(s) for s in split_sizes]
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.split_sizes, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            "split_sizes": self.split_sizes,
            "axis": self.axis,
        })
        return config

