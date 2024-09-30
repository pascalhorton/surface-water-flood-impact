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
        if not self.options.combined_transformer:
            input_daily = layers.Input(
                shape=(self.input_daily_prec_size,),
                name='input_daily')
            input_high_freq = layers.Input(
                shape=(self.input_high_freq_prec_size,),
                name='input_high_freq')
            input_attributes = layers.Input(
                shape=(self.input_attributes_size,),
                name='input_attributes')

            # Transformer for daily precipitation
            x_daily_pos = self.get_positional_encoding(self.input_daily_prec_size)
            x_daily = input_daily + x_daily_pos
            # Project the input into the model dimension
            x_daily = layers.Dense(
                self.options.tx_model_dim_daily,
                name='dense_tx_proj_daily'
            )(x_daily)
            for _ in range(self.options.nb_transformer_blocks_daily):
                x_daily = self.transformer_block(
                    x_daily,
                    model_dim=self.options.tx_model_dim_daily,
                    num_heads=self.options.num_heads_daily,
                    ff_dim=self.options.ff_dim_daily,
                    dropout_rate=self.options.dropout_rate_daily,
                    use_cnn=self.options.use_cnn_in_tx)

            # Transformer for high-frequency precipitation
            x_high_freq_pos = self.get_positional_encoding(self.input_high_freq_prec_size)
            x_high_freq = input_high_freq + x_high_freq_pos
            # Project the input into the model dimension
            x_high_freq = layers.Dense(
                self.options.tx_model_dim_high_freq,
                name='dense_tx_proj_high_freq'
            )(x_high_freq)
            for _ in range(self.options.nb_transformer_blocks_high_freq):
                x_high_freq = self.transformer_block(
                    x_high_freq,
                    model_dim=self.options.tx_model_dim_high_freq,
                    num_heads=self.options.num_heads_high_freq,
                    ff_dim=self.options.ff_dim_high_freq,
                    dropout_rate=self.options.dropout_rate_high_freq,
                    use_cnn=self.options.use_cnn_in_tx)

            # Transformer for attributes
            x_attributes = input_attributes
            # Project the input into the model dimension
            x_attributes = layers.Dense(
                self.options.tx_model_dim_attributes,
                name='dense_tx_proj_attributes'
            )(x_attributes)
            for _ in range(self.options.nb_transformer_blocks_attributes):
                x_attributes = self.transformer_block(
                    x_attributes,
                    model_dim=self.options.tx_model_dim_attributes,
                    num_heads=self.options.num_heads_attributes,
                    ff_dim=self.options.ff_dim_attributes,
                    dropout_rate=self.options.dropout_rate_attributes,
                    use_cnn=False)

            # Concatenate
            x = layers.Concatenate(axis=-1)([x_daily, x_high_freq, x_attributes])

        else:
            max_length = max(self.input_daily_prec_size,
                             self.input_high_freq_prec_size)

            input_daily = layers.Input(
                shape=(max_length,),
                name='input_daily')
            input_high_freq = layers.Input(
                shape=(max_length,),
                name='input_high_freq')
            input_attributes = layers.Input(
                shape=(self.input_attributes_size,),
                name='input_attributes')

            x_daily_pos = self.get_positional_encoding(self.input_daily_prec_size)
            x_daily = input_daily + x_daily_pos
            x_daily = layers.Dense(
                self.options.tx_model_dim_daily,
                name='dense_tx_proj_daily'
            )(x_daily)

            x_high_freq_pos = self.get_positional_encoding(self.input_high_freq_prec_size)
            x_high_freq = input_high_freq + x_high_freq_pos
            x_high_freq = layers.Dense(
                self.options.tx_model_dim_high_freq,
                name='dense_tx_proj_high_freq'
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

            for _ in range(self.options.nb_transformer_blocks_combined):
                x = self.transformer_block(
                    x,
                    model_dim=self.options.tx_model_dim_combined,
                    num_heads=self.options.num_heads_combined,
                    ff_dim=self.options.ff_dim_combined,
                    dropout_rate=self.options.dropout_rate_combined,
                    use_cnn=self.options.use_cnn_in_tx)

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

    @staticmethod
    def get_positional_encoding(seq_len):
        """
        Get the positional encoding.

        Parameters
        ----------
        seq_len: int
            The sequence length.

        Returns
        -------
        The positional encoding.
        """
        d_model = 1

        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

        # Apply sin to even indices and cos to odd indices
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])  # Even
        pos_encoding[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])  # Odd

        # Squeeze as d_model is 1
        pos_encoding = pos_encoding.squeeze()

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def transformer_block(self, inputs, model_dim=512, num_heads=8, ff_dim=128,
                          dropout_rate=0.1, use_cnn=False):
        """
        Transformer encoder.

        Parameters
        ----------
        inputs: tensor
            The input tensor.
        num_heads: int
            The number of heads.
        model_dim: int
            The model dimension.
        ff_dim: int
            The feed-forward dimension.
        dropout_rate: float
            The dropout rate.
        use_cnn: bool
            Whether to use a CNN or not. If not, a dense layer is used.

        Returns
        -------
        The output tensor.
        """
        key_dim = model_dim // num_heads

        # Self-attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = layers.Dropout(dropout_rate)(x)
        res = layers.Add()([inputs, x])

        # Feed-forward network
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        if use_cnn:
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Conv1D(filters=model_dim, kernel_size=1)(x)
        else:
            layer_name = f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
            x = layers.Dense(ff_dim, activation='relu', name=layer_name)(x)
            layer_name = f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
            x = layers.Dense(model_dim, name=layer_name)(x)
            x = layers.Dropout(dropout_rate)(x)

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
