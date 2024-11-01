"""
Class for the Transformer model.
"""

from keras import layers, models
import tensorflow as tf
import numpy as np


class ModelTransformer(models.Model):
    """
    Transformer model factory.

    Parameters
    ----------
    task: str
        The task. Options are: 'regression', 'classification'
    options: ImpactTransformerOptions
        The options.
    input_daily_prec_size: int, None
        The size of the daily precipitation data.
    input_high_freq_prec_size: int, None
        The size of the high-frequency precipitation data.
    input_attributes_size: int, None
        The input 1D size.
    """

    def __init__(self, task, options, input_daily_prec_size, input_high_freq_prec_size,
                 input_attributes_size):
        super(ModelTransformer, self).__init__()
        self.model = None
        self.task = task
        self.options = options
        self.input_daily_prec_size = input_daily_prec_size
        self.input_high_freq_prec_size = input_high_freq_prec_size
        self.input_attributes_size = input_attributes_size

        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        self._build_model()

    def _build_model(self):
        """
        Build the model.
        """
        input_daily = layers.Input(
            shape=(self.input_daily_prec_size, 1),
            name='input_daily')
        input_high_freq = layers.Input(
            shape=(self.input_high_freq_prec_size, 1),
            name='input_high_freq')
        input_attributes = layers.Input(
            shape=(self.input_attributes_size, 1),
            name='input_attributes')

        if not self.options.combined_transformer:
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

        else:
            x_daily = self.project_to_model_dim(input_daily)
            x_high_freq = self.project_to_model_dim(input_high_freq)



            x_daily = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_daily)

            x_high_freq = AddFixedPositionalEmbedding(
                self.options.tx_model_dim
            )(x_high_freq)

            x = layers.Concatenate(axis=1)([x_daily, x_high_freq])
            # Broadcast static attributes across timesteps
            x_attributes = tf.expand_dims(
                input_attributes,
                axis=1)  # Shape becomes (batch_size, 1, num_static_features)
            x_attributes = tf.tile(
                x_attributes,
                [1, max_length, 1])  # Broadcast static features across timesteps

            # Combine time series and static attributes into a single sequence
            x = layers.Concatenate(axis=-1)([x, x_attributes])

            for _ in range(self.options.nb_transformer_blocks):
                x = self.transformer_block(x, use_cnn=self.options.use_cnn_in_tx)

        # Flatten
        x = layers.Flatten()(x)

        # Fully connected
        for i in range(self.options.nb_dense_layers):
            if self.options.nb_dense_units_decreasing:
                nb_units = self.options.nb_dense_units // (2 ** i)
            else:
                nb_units = self.options.nb_dense_units
            x = layers.Dense(nb_units, activation=self.options.inner_activation_dense,
                             name=f'dense_ffn_{i}')(x)

            if self.options.with_batchnorm_dense:
                x = layers.BatchNormalization(name=f'batchnorm_dense_{i}')(x)

            if self.options.dropout_rate_dense > 0:
                x = layers.Dropout(rate=self.options.dropout_rate_dense,
                                   name=f'dropout_dense_{i}')(x)

        # Last activation
        output = layers.Dense(1, activation=self.last_activation,
                              name=f'dense_last')(x)

        # Build model
        self.model = models.Model(
            inputs=[input_daily, input_high_freq, input_attributes],
            outputs=output)

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
            self.options.tx_model_dim,
            name=f'dense_proj_{int(1e4 * tf.random.uniform([]))}',
            activation=self.options.embeddings_activation
        )(inputs)

        if self.options.embeddings_2_layers:
            x = layers.Dense(
                self.options.tx_model_dim,
                name=f'dense_proj_{int(1e4 * tf.random.uniform([]))}',
                activation=self.options.embeddings_activation
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
        res = inputs + x

        # Feed-forward network
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        if use_cnn:
            x = layers.Conv1D(
                filters=self.options.ff_dim,
                kernel_size=1,
                activation='relu'
            )(x)
            x = layers.Dropout(self.options.dropout_rate)(x)
            x = layers.Conv1D(
                filters=self.options.tx_model_dim,
                kernel_size=1
            )(x)
        else:
            x = layers.Dense(
                self.options.ff_dim,
                activation='relu',
                name=f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
            )(x)
            x = layers.Dense(
                self.options.tx_model_dim,
                name=f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
            )(x)
            x = layers.Dropout(self.options.dropout_rate)(x)

        x = layers.Add()([x, res])

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


class AddFixedPositionalEmbedding(layers.Layer):
    """
    Positional embedding layer.
    Source: https://pylessons.com/transformers-introduction
    """
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.pos_encoding = self.get_positional_encoding()

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
        length = x.shape[1]
        pos = self.pos_encoding[tf.newaxis, :length, :]
        assert pos.shape[1:2] == x.shape[1:2]
        x = x + pos

        return x
