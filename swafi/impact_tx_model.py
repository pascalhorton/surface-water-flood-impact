"""
Class for the Transformer model.
"""

from keras import layers, models
import tensorflow as tf


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
            x_daily = input_daily
            for _ in range(self.options.nb_transformer_blocks_daily):
                x_daily = self.transformer_block(
                    x_daily,
                    num_heads=self.options.num_heads_daily,
                    ff_dim=self.options.ff_dim_daily,
                    dropout_rate=self.options.dropout_rate_daily,
                    activation=self.options.inner_activation_tx)

            # Transformer for high-frequency precipitation
            x_high_freq = input_high_freq
            for _ in range(self.options.nb_transformer_blocks_high_freq):
                x_high_freq = self.transformer_block(
                    x_high_freq,
                    num_heads=self.options.num_heads_high_freq,
                    ff_dim=self.options.ff_dim_high_freq,
                    dropout_rate=self.options.dropout_rate_high_freq,
                    activation=self.options.inner_activation_tx)

            # Transformer for attributes
            x_attributes = input_attributes
            for _ in range(self.options.nb_transformer_blocks_attributes):
                x_attributes = self.transformer_block(
                    x_attributes,
                    num_heads=self.options.num_heads_attributes,
                    ff_dim=self.options.ff_dim_attributes,
                    dropout_rate=self.options.dropout_rate_attributes,
                    activation=self.options.inner_activation_tx)

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

            x = layers.Concatenate(axis=1)([input_daily, input_high_freq])
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
                    num_heads=self.options.num_heads_combined,
                    ff_dim=self.options.ff_dim_combined,
                    dropout_rate=self.options.dropout_rate_combined,
                    activation=self.options.inner_activation_tx)

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
        self.model = models.Model(
            inputs=[input_daily, input_high_freq, input_attributes],
            outputs=output)

    @staticmethod
    def transformer_block(x_in, num_heads=4, ff_dim=128, dropout_rate=0.1,
                          activation='relu'):
        """
        Transformer encoder.

        Parameters
        ----------
        x_in: tensor
            The input tensor.
        num_heads: int
            The number of heads.
        ff_dim: int
            The feed-forward dimension.
        dropout_rate: float
            The dropout rate.
        activation: str
            The activation function.

        Returns
        -------
        The output tensor.
        """
        # Self-attention
        x_attn = layers.MultiHeadAttention(num_heads,x_in.shape[-1])(x_in, x_in)
        x_attn = layers.Dropout(dropout_rate)(x_attn)
        x_attn = layers.LayerNormalization()(x_in + x_attn)

        # Feed-forward network
        x_ffn = layers.Dense(ff_dim, activation=activation,
                             name=f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
                             )(x_attn)
        x_ffn = layers.Dropout(dropout_rate)(x_ffn)
        x_ffn = layers.Dense(x_attn.shape[1], activation=activation,
                             name=f'dense_tx_{int(1e4 * tf.random.uniform([]))}'
                             )(x_ffn)
        x_ffn = layers.LayerNormalization()(x_attn + x_ffn)

        return x_ffn

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
