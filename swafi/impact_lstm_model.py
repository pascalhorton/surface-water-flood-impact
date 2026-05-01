"""
Keras model definition for LSTM + multi-head self-attention.
"""
import logging

import keras
import numpy as np


logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="swafi")
class ModelLstm(keras.models.Model):
    """
    LSTM + multi-head self-attention model factory.

    The precipitation time-series input has shape (T, 1, 1, C) — same format
    as the CNN data generator with precip_window_size=1. It is reshaped to (T, C)
    before the LSTM layers. Optional static tabular features are concatenated after
    temporal pooling.

    Architecture:
        Input_3D (T, 1, 1, C) → Reshape (T, C) → Dense projection →
        LSTM × N (return_sequences=True) →
        MultiHeadAttention (self-attention) + LayerNorm (residual) →
        GlobalAveragePooling1D →
        [Concatenate static features if present] →
        Dense head × M → sigmoid output

    Parameters
    ----------
    task: str
        'classification' or 'regression'.
    options: ImpactLstmOptions
        Model options.
    input_3d_size: list|None
        Shape (T, 1, 1, C) of the precipitation input.
    input_1d_size: list|None
        Shape (F,) of the static tabular input.
    output_bias_init: float
        Initial value for the output layer bias (log-odds of class prior).
    """

    def __init__(self, trainable=True, dtype=None, task='classification',
                 options=None, input_3d_size=None, input_1d_size=None,
                 output_bias_init=0.0, *args, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, *args, **kwargs)
        self.model = None
        self.task = task
        self.options = options
        self.input_3d_size = list(input_3d_size) if input_3d_size is not None else None
        self.input_1d_size = list(input_1d_size) if input_1d_size is not None else None
        self.output_bias_init = float(output_bias_init)
        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        # Training-set statistics stored for inference-time consistency.
        self.mean_static = None
        self.std_static = None
        self.min_static = None
        self.max_static = None
        self.mean_precip = None
        self.std_precip = None
        self.q99_precip = None

    def set_feature_stats(self, mean_static=None, std_static=None,
                          min_static=None, max_static=None,
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
        base_config = super().get_config()
        try:
            options_cfg = (keras.saving.serialize_keras_object(self.options)
                           if self.options is not None else None)
        except Exception:
            options_cfg = getattr(self.options, 'get_config', lambda: None)()

        config = {
            "task": self.task,
            "options": options_cfg,
            "input_3d_size": self.input_3d_size,
            "input_1d_size": self.input_1d_size,
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
        trainable = config.get("trainable", True)
        dtype = config.get("dtype", None)

        options_cfg = config.get("options", None)
        options = None
        if options_cfg is not None:
            try:
                options = keras.saving.deserialize_keras_object(options_cfg)
            except Exception:
                options = None

        instance = cls(trainable=trainable, dtype=dtype)
        instance.task = config.get("task", "classification")
        instance.options = options
        instance.input_3d_size = config.get("input_3d_size", None)
        instance.input_1d_size = config.get("input_1d_size", None)
        instance.last_activation = 'relu' if instance.task == 'regression' else 'sigmoid'
        instance.output_bias_init = config.get("output_bias_init", 0.0)

        instance.mean_static = cls._deserialize_array(config.get("mean_static", None))
        instance.std_static = cls._deserialize_array(config.get("std_static", None))
        instance.min_static = cls._deserialize_array(config.get("min_static", None))
        instance.max_static = cls._deserialize_array(config.get("max_static", None))
        instance.mean_precip = cls._deserialize_array(config.get("mean_precip", None))
        instance.std_precip = cls._deserialize_array(config.get("std_precip", None))
        instance.q99_precip = cls._deserialize_array(config.get("q99_precip", None))

        build_cfg = config.get("build_config", None)
        if build_cfg is not None:
            try:
                instance.model = keras.models.Model.from_config(build_cfg)
            except Exception:
                instance.model = None

        return instance

    def get_build_config(self):
        if self.model is None:
            return None
        return self.model.get_config()

    def build_model(self, options=None):
        """
        Build the Keras functional model.

        Parameters
        ----------
        options: ImpactLstmOptions|None
            If provided, overrides self.options.
        """
        if options is not None:
            self.options = options

        if self.input_3d_size is None and self.input_1d_size is None:
            raise ValueError("At least one input size must be provided")

        x = None

        if self.input_3d_size is not None:
            # Input shape: (T, 1, 1, C)
            t_len = self.input_3d_size[0]
            n_channels = self.input_3d_size[3]

            input_3d = keras.layers.Input(shape=self.input_3d_size, name='input_3d')

            # Squeeze spatial dims → (T, C)
            x = keras.layers.Reshape((t_len, n_channels), name='reshape_1px')(input_3d)

            # Project to LSTM hidden dimension
            x = keras.layers.Dense(self.options.lstm_units, name='lstm_proj')(x)

            # Stacked LSTM layers — forward-only (causal)
            for i in range(self.options.lstm_nb_layers):
                x = keras.layers.LSTM(
                    self.options.lstm_units,
                    return_sequences=True,
                    name=f'lstm_{i}'
                )(x)
                if self.options.dropout_rate_lstm > 0:
                    x = keras.layers.Dropout(
                        self.options.dropout_rate_lstm, name=f'lstm_drop_{i}'
                    )(x)

            # Multi-head self-attention on all LSTM output states
            # Non-causal: full event window is available at inference time
            attn_out = keras.layers.MultiHeadAttention(
                num_heads=self.options.attention_heads,
                key_dim=self.options.attention_key_dim,
                name='mha'
            )(x, x)
            x = keras.layers.LayerNormalization(name='mha_ln')(x + attn_out)

            # Pool temporal dimension → (lstm_units,)
            x = keras.layers.GlobalAveragePooling1D(name='temporal_avg')(x)

        if self.input_1d_size is not None:
            input_1d = keras.layers.Input(shape=self.input_1d_size, name='input_1d')

            if self.input_3d_size is not None:
                x = keras.layers.Concatenate(name='concat_1d')([x, input_1d])
            else:
                x = input_1d

        # Dense head (mirrors ModelCnn pattern)
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = max(self.options.nb_dense_units // (2 ** i), 4)
            else:
                nb_units = self.options.nb_dense_units

            x_skip = x

            x = keras.layers.Dense(nb_units, name=f'dense_{i}')(x)

            if getattr(self.options, 'use_layernorm_dense', False):
                x = keras.layers.LayerNormalization(name=f'layernorm_dense_{i}')(x)
            elif self.options.use_batchnorm_dense:
                x = keras.layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            x = keras.layers.Activation(
                self.options.inner_activation_dense, name=f'act_dense_{i}'
            )(x)

            if self.options.dropout_rate_dense > 0:
                x = keras.layers.Dropout(
                    rate=self.options.dropout_rate_dense, name=f'dropout_dense_{i}'
                )(x)

            if getattr(self.options, 'use_residual_dense', False):
                if x_skip.shape[-1] == nb_units:
                    x = keras.layers.Add(name=f'res_dense_{i}')([x, x_skip])
                else:
                    x_proj = keras.layers.Dense(
                        nb_units, use_bias=False, name=f'res_proj_{i}'
                    )(x_skip)
                    x = keras.layers.Add(name=f'res_dense_{i}')([x, x_proj])

        output = keras.layers.Dense(
            1,
            activation=self.last_activation,
            bias_initializer=keras.initializers.Constant(self.output_bias_init),
            name='dense_last'
        )(x)

        if self.input_3d_size is not None and self.input_1d_size is not None:
            self.model = keras.models.Model(inputs=[input_3d, input_1d], outputs=output)
        elif self.input_3d_size is None:
            self.model = keras.models.Model(inputs=input_1d, outputs=output)
        else:
            self.model = keras.models.Model(inputs=input_3d, outputs=output)

    def call(self, inputs, training=None, **kwargs):
        if self.model is None:
            raise ValueError("Model not built — call build_model() first")
        return self.model(inputs, training=training, **kwargs)
