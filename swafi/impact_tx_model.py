"""
Class for the Transformer model.
"""

import logging

from keras import layers, models
import keras
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="swafi")
class ModelTransformer(models.Model):
    """
    Transformer model factory.

    Parameters
    ----------
    trainable: bool
        Whether the model is trainable.
    dtype: str|None
        The data type.
    task: str
        The task. Options are: 'regression', 'classification'
    options: ImpactTransformerOptions|None
        The options.
    input_daily_prec_size: int, None
        The size of the daily precipitation data.
    input_high_freq_prec_size: int, None
        The size of the high-frequency precipitation data.
    input_attributes_size: int, None
        The input 1D size.
    output_bias_init: float
        The bias initialisation of the output layer.
    **kwargs
        Additional arguments to pass to keras.models.Model.
    """

    def __init__(self, trainable=True, dtype=None, task='classification', options=None,
                 input_daily_prec_size=None, input_high_freq_prec_size=None,
                 input_attributes_size=None, output_bias_init=0.0, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.model = None
        self.task = task
        self.options = options
        self.input_daily_prec_size = input_daily_prec_size
        self.input_high_freq_prec_size = input_high_freq_prec_size
        self.input_attributes_size = input_attributes_size
        self.output_bias_init = float(output_bias_init)

        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        # Deserialization creates an empty shell and restores the inner graph from
        # its saved config instead (see from_config).
        if options is not None and input_attributes_size is not None:
            self._build_model()

    def get_config(self):
        """
        Return a serializable config for this wrapper.
        """
        try:
            base_config = super().get_config()
        except NotImplementedError:
            base_config = {"name": self.name, "trainable": self.trainable}

        options_cfg = (keras.saving.serialize_keras_object(self.options)
                       if self.options is not None else None)

        config = {
            "task": self.task,
            "options": options_cfg,
            "input_daily_prec_size": self.input_daily_prec_size,
            "input_high_freq_prec_size": self.input_high_freq_prec_size,
            "input_attributes_size": self.input_attributes_size,
            "output_bias_init": self.output_bias_init,
            # The projection layers get random names, so the inner graph has to be
            # restored from its own config: rebuilding it from the options would
            # give layer names that no longer match the saved weights.
            "inner_model_config": (self.model.get_config()
                                   if self.model is not None else None),
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Recreate the wrapper and restore the internal Keras model.
        Keras will call this when deserializing the custom object.
        """
        options_cfg = config.get("options", None)
        options = None
        if options_cfg is not None:
            try:
                options = keras.saving.deserialize_keras_object(options_cfg)
            except Exception:
                logger.warning("Could not deserialize the Transformer options.")

        # Create an empty shell: the inner model is restored below, not rebuilt.
        instance = cls(trainable=config.get("trainable", True),
                       dtype=config.get("dtype", None))
        instance.task = config.get("task", 'classification')
        instance.options = options
        instance.input_daily_prec_size = config.get("input_daily_prec_size", None)
        instance.input_high_freq_prec_size = config.get(
            "input_high_freq_prec_size", None)
        instance.input_attributes_size = config.get("input_attributes_size", None)
        instance.output_bias_init = config.get("output_bias_init", 0.0)
        instance.last_activation = (
            'relu' if instance.task == 'regression' else 'sigmoid')

        inner_config = config.get("inner_model_config", None)
        if inner_config is not None:
            instance.model = models.Model.from_config(inner_config)
        elif options is not None and instance.input_attributes_size is not None:
            instance._build_model()

        return instance

    def _build_model(self):
        """
        Build the model.
        """
        input_daily = layers.Input(
            shape=(self.input_daily_prec_size,),
            name='input_daily')
        input_high_freq = layers.Input(
            shape=(self.input_high_freq_prec_size,),
            name='input_high_freq')
        input_attributes = layers.Input(
            shape=(self.input_attributes_size,),
            name='input_attributes')

        input_daily = keras.ops.expand_dims(input_daily, axis=-1)
        input_high_freq = keras.ops.expand_dims(input_high_freq, axis=-1)

        if self.options.architecture == 'separate':
            input_attributes = keras.ops.expand_dims(input_attributes, axis=-1)

            x_daily = self.project_to_model_dim(input_daily)
            x_daily = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_daily)

            for _ in range(self.options.nb_transformer_blocks):
                x_daily = self.transformer_block(
                    x_daily, use_cnn=self.options.use_cnn_in_tx)

            x_high_freq = self.project_to_model_dim(input_high_freq)
            x_high_freq = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_high_freq)

            for _ in range(self.options.nb_transformer_blocks):
                x_high_freq = self.transformer_block(
                    x_high_freq, use_cnn=self.options.use_cnn_in_tx)

            # Project the attributes input into the model dimension
            x_attributes = self.project_to_model_dim(input_attributes)

            for _ in range(self.options.nb_transformer_blocks):
                x_attributes = self.transformer_block(x_attributes, use_cnn=False)

            # Concatenate
            x = layers.Concatenate(axis=1)([x_daily, x_high_freq, x_attributes])

        elif self.options.architecture == 'combined_fixed_embeddings':
            # Project and concatenate the precipitation inputs
            x_daily = self.project_to_model_dim(input_daily)
            x_high_freq = self.project_to_model_dim(input_high_freq)
            x = layers.Concatenate(axis=1)([x_daily, x_high_freq])
            x = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x)

            # Project the attributes input into the model dimension
            x_attributes = self.project_to_model_dim(input_attributes)
            x_attributes = keras.ops.expand_dims(x_attributes, axis=1)

            # Combine time series and static attributes into a single sequence
            x = layers.Concatenate(axis=1)([x, x_attributes])

            for _ in range(self.options.nb_transformer_blocks):
                x = self.transformer_block(x, use_cnn=self.options.use_cnn_in_tx)

        elif self.options.architecture == 'combined_broadcast':
            # Project and concatenate the precipitation inputs
            x_daily = self.project_to_model_dim(input_daily)
            x_high_freq = self.project_to_model_dim(input_high_freq)
            x = layers.Concatenate(axis=1)([x_daily, x_high_freq])
            x = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x)

            x_attributes = self.project_to_model_dim(input_attributes)
            x_attributes = keras.ops.expand_dims(x_attributes, axis=1)

            # Add the attributes to the time series
            x = layers.Add()([x, x_attributes])

            for _ in range(self.options.nb_transformer_blocks):
                x = self.transformer_block(x, use_cnn=self.options.use_cnn_in_tx)

        elif self.options.architecture == 'combined_learned_embeddings':
            embeddings_activation = self.options.embeddings_activation
            if embeddings_activation == 'None':
                embeddings_activation = None

            # Project and concatenate the precipitation inputs
            x_daily = self.project_to_model_dim(input_daily)
            x_high_freq = self.project_to_model_dim(input_high_freq)
            x = layers.Concatenate(axis=1)([x_daily, x_high_freq])

            # Add positional embeddings
            x = AddLearnedPositionalEmbedding(
                model_dim=self.options.tx_model_dim,
                daily_prec_size=self.input_daily_prec_size,
                high_freq_prec_size=self.input_high_freq_prec_size,
                embeddings_activation=embeddings_activation,
                embeddings_2_layers=self.options.embeddings_2_layers,
                use_flag_embedding=self.options.use_precip_type_embedding
            )(x)

            # Project the attributes input into the model dimension
            if self.options.use_single_attributes_vector:
                x_attributes = layers.Dense(
                    self.options.tx_model_dim,
                    name=f'dense_proj_{int(1e6 * np.random.uniform())}',
                    activation=embeddings_activation
                )(input_attributes)

                if self.options.embeddings_2_layers:
                    x_attributes = layers.Dense(
                        self.options.tx_model_dim,
                        name=f'dense_proj_{int(1e6 * np.random.uniform())}',
                        activation=embeddings_activation
                    )(x_attributes)

                x_attributes = keras.ops.expand_dims(x_attributes, axis=1)
            else:
                x_attributes = self.project_to_model_dim(input_attributes)

            # Combine time series and static attributes into a single sequence
            x = layers.Concatenate(axis=1)([x, x_attributes])

            for _ in range(self.options.nb_transformer_blocks):
                x = self.transformer_block(x, use_cnn=False)

        elif self.options.architecture == 'hybrid':
            # Separate processing
            x_daily = self.project_to_model_dim(input_daily)
            x_daily = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_daily)
            for _ in range(self.options.nb_transformer_blocks):
                x_daily = self.transformer_block(
                    x_daily, use_cnn=self.options.use_cnn_in_tx)

            x_high_freq = self.project_to_model_dim(input_high_freq)
            x_high_freq = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_high_freq)
            for _ in range(self.options.nb_transformer_blocks):
                x_high_freq = self.transformer_block(
                    x_high_freq, use_cnn=self.options.use_cnn_in_tx)

            # Concatenate and further processing
            x = layers.Concatenate(axis=1)([x_daily, x_high_freq])
            x = AddFixedPositionalEmbedding(self.options.tx_model_dim)(x)

            x_attributes = self.project_to_model_dim(input_attributes)
            x_attributes = keras.ops.expand_dims(x_attributes, axis=1)
            x = layers.Add()([x, x_attributes])

            for _ in range(self.options.nb_transformer_blocks):
                x = self.transformer_block(x, use_cnn=self.options.use_cnn_in_tx)

        else:
            raise ValueError(f'Architecture {self.options.architecture} not supported.')

        # Flatten
        x = layers.Flatten()(x)

        # Fully connected
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = self.options.nb_dense_units // (2 ** i)
                # Keep at least 4 units
                nb_units = max(nb_units, 4)
            else:
                nb_units = self.options.nb_dense_units
            x = layers.Dense(nb_units, activation=self.options.inner_activation_dense,
                             name=f'dense_ffn_{i}')(x)

            if self.options.use_batchnorm_dense:
                x = layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            if self.options.dropout_rate_dense > 0:
                x = layers.Dropout(rate=self.options.dropout_rate_dense,
                                   name=f'dropout_dense_{i}')(x)

        # Last activation — bias initialized to log-odds of class prior for faster convergence
        output = layers.Dense(
            1,
            activation=self.last_activation,
            bias_initializer=keras.initializers.Constant(self.output_bias_init),
            name='dense_last'
        )(x)

        # Build model
        self.model = models.Model(
            inputs=[input_daily, input_high_freq, input_attributes],
            outputs=output)

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

    def project_to_model_dim(self, inputs):
        """
        Project the input into the model dimension.

        Parameters
        ----------
        inputs: tensor
            The input tensor.

        Returns
        -------
        The output tensor.
        """
        embeddings_activation = self.options.embeddings_activation
        if embeddings_activation == 'None':
            embeddings_activation = None

        x = layers.Dense(
            self.options.tx_model_dim,
            name=f'dense_proj_{int(1e6 * np.random.uniform())}',
            activation=embeddings_activation
        )(inputs)

        if self.options.embeddings_2_layers:
            x = layers.Dense(
                self.options.tx_model_dim,
                name=f'dense_proj_{int(1e6 * np.random.uniform())}',
                activation=embeddings_activation
            )(x)

        return x

    def transformer_block(self, inputs, use_cnn=False):
        """
        Transformer encoder.

        Parameters
        ----------
        inputs: tensor
            The input tensor.
        use_cnn: bool
            Whether to use a CNN or not. If not, a dense layer is used.

        Returns
        -------
        The output tensor.
        """
        # Check if model_dim is divisible by num_heads
        assert self.options.tx_model_dim % self.options.num_heads == 0
        key_dim = self.options.tx_model_dim // self.options.num_heads

        # Self-attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            num_heads=self.options.num_heads,
            key_dim=key_dim
        )(x, x)
        x = layers.Dropout(self.options.dropout_rate)(x)
        res = layers.Add()([inputs, x])

        # Feed-forward network
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        if use_cnn:
            x = layers.Conv1D(
                filters=self.options.ff_dim,
                kernel_size=1,
                activation=self.options.inner_activation_tx
            )(x)
            x = layers.Dropout(self.options.dropout_rate)(x)
            x = layers.Conv1D(
                filters=self.options.tx_model_dim,
                kernel_size=1
            )(x)
        else:
            x = layers.Dense(
                self.options.ff_dim,
                activation=self.options.inner_activation_tx,
                name=f'dense_tx_{int(1e6 * np.random.uniform())}'
            )(x)
            x = layers.Dense(
                self.options.tx_model_dim,
                name=f'dense_tx_{int(1e6 * np.random.uniform())}'
            )(x)
            x = layers.Dropout(self.options.dropout_rate)(x)

        x = layers.Add()([x, res])

        return x



@keras.saving.register_keras_serializable(package="swafi")
class AddLearnedPositionalEmbedding(layers.Layer):
    """
    learned positional embedding layer.
    """
    def __init__(self, model_dim, daily_prec_size, high_freq_prec_size,
                 embeddings_activation=None, embeddings_2_layers=False,
                 use_flag_embedding=True, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.daily_prec_size = daily_prec_size
        self.high_freq_prec_size = high_freq_prec_size
        self.embeddings_activation = embeddings_activation
        self.embeddings_2_layers = embeddings_2_layers
        self.use_flag_embedding = use_flag_embedding

        # The embeddings come out of randomly initialized projections, so they are
        # kept as (non-trainable) weights: recomputing them when the model is
        # reloaded would give a different embedding than the one trained with.
        self.temporal_embedding = self._as_weight(
            self.get_temporal_embedding(), 'temporal_embedding')
        self.flag_embedding = None
        if self.use_flag_embedding:
            self.flag_embedding = self._as_weight(
                self.get_flag_embedding(), 'flag_embedding')

    def _as_weight(self, value, name):
        """
        Store a constant tensor as a non-trainable weight so that it is saved
        along with the model.

        Parameters
        ----------
        value: tensor
            The constant tensor to store.
        name: str
            The name of the weight.

        Returns
        -------
        The corresponding non-trainable weight.
        """
        value = tf.convert_to_tensor(value, dtype=tf.float32)

        return self.add_weight(
            shape=tuple(value.shape),
            name=name,
            trainable=False,
            initializer=lambda shape, dtype=None: tf.cast(
                value, dtype if dtype is not None else tf.float32),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.model_dim,
            "daily_prec_size": self.daily_prec_size,
            "high_freq_prec_size": self.high_freq_prec_size,
            "embeddings_activation": self.embeddings_activation,
            "embeddings_2_layers": self.embeddings_2_layers,
            "use_flag_embedding": self.use_flag_embedding,
        })
        return config

    def get_temporal_embedding(self):
        """
        Get the temporal (positional) embedding. This allows the model to learn
        the position of the input data (in time).

        Returns
        -------
        The temporal embedding.
        """
        l_dims = self.daily_prec_size + self.high_freq_prec_size
        t_emb = np.arange(l_dims) / l_dims
        t_emb = np.expand_dims(t_emb, axis=-1)
        t_emb = tf.convert_to_tensor(t_emb, dtype=tf.float32)
        t_emb = self.project_to_model_dim(t_emb)
        t_emb = tf.expand_dims(t_emb, axis=0)

        return t_emb

    def get_flag_embedding(self):
        """
        Get the flag embedding. This allows the model to learn the difference
        between daily and high-frequency precipitation data.

        Returns
        -------
        The flag embedding.
        """
        flag_daily = np.ones((self.daily_prec_size,))
        flag_hourly = np.zeros((self.high_freq_prec_size,))
        flags = np.concatenate([flag_daily, flag_hourly], axis=0)
        flags = tf.convert_to_tensor(flags, dtype=tf.int32)
        flags = layers.Embedding(
            input_dim=2,
            output_dim=self.model_dim
        )(flags)
        flags = tf.expand_dims(flags, axis=0)

        return flags

    def project_to_model_dim(self, inputs):
        """
        Project the input into the model dimension.

        Parameters
        ----------
        inputs: tensor
            The input tensor.

        Returns
        -------
        The output tensor.
        """
        x = layers.Dense(
            self.model_dim,
            name=f'dense_proj_{int(1e6 * np.random.uniform())}',
            activation=self.embeddings_activation
        )(inputs)

        if self.embeddings_2_layers:
            x = layers.Dense(
                self.model_dim,
                name=f'dense_proj_{int(1e6 * np.random.uniform())}',
                activation=self.embeddings_activation
            )(x)

        return x

    def call(self, x):
        """
        Call the layer.

        Parameters
        ----------
        x: tensor
            The input tensor.

        Returns
        -------
        The output tensor.
        """
        t_emb = self.temporal_embedding

        if self.use_flag_embedding:
            flags = self.flag_embedding
            embedding = t_emb + flags
        else:
            embedding = t_emb

        target_shape = tf.shape(x)
        embedding = tf.broadcast_to(embedding, target_shape)

        return layers.Add()([x, embedding])


@keras.saving.register_keras_serializable(package="swafi")
class AddFixedPositionalEmbedding(layers.Layer):
    """
    Positional embedding layer.
    Source: https://pylessons.com/transformers-introduction
    """
    def __init__(self, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        # Deterministic encoding: recomputing it on load gives the same values.
        self.pos_encoding = self.get_positional_encoding()

    def get_config(self):
        config = super().get_config()
        config.update({"model_dim": self.model_dim})
        return config

    def get_positional_encoding(self, length=1024):
        """
        Get the positional encoding.

        Parameters
        ----------
        length: int
            The sequence length.

        Returns
        -------
        The positional encoding.
        """
        # Create the positional encoding
        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / self.model_dim) for i in
             range(self.model_dim)]
            if pos != 0 else np.zeros(self.model_dim) for pos in range(length)])

        # Apply sine to even indices in the array; 2i
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # Apply cosine to odd indices in the array; 2i+1
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        return tf.cast(position_enc, dtype=tf.float32)

    def call(self, x):
        """
        Call the layer.

        Parameters
        ----------
        x: tensor
            The input tensor.

        Returns
        -------
        The output tensor.
        """
        length = x.shape[1]
        pos = self.pos_encoding[tf.newaxis, :length, :]
        assert pos.shape[1:2] == x.shape[1:2]

        target_shape = tf.shape(x)
        pos = tf.broadcast_to(pos, target_shape)

        return layers.Add()([x, pos])
