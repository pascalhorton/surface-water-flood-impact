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
    input_daily_prec_size: list, None
        The size of the daily precipitation data.
    input_high_freq_prec_size: list, None
        The size of the high-frequency precipitation data.
    input_attributes_size: list, None
        The input 1D size.
    """

    def __init__(self, task, options, input_daily_prec_size, input_high_freq_prec_size,
                 input_attributes_size):
        super(ModelTransformer, self).__init__()
        self.model = None
        self.task = task
        self.options = options
        self.input_daily_prec_size = list(input_daily_prec_size)
        self.input_high_freq_prec_size = list(input_high_freq_prec_size)
        self.input_attributes_size = list(input_attributes_size)

        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        self._check_input_size()
        self._build_model()

    def _check_input_size(self):
        """
        Check the input size.
        """
        assert len(self.input_daily_prec_size) == 2
        assert len(self.input_high_freq_prec_size) == 2
        assert len(self.input_attributes_size) == 1

    def _build_model(self):
        """
        Build the model.
        """
        input_daily = layers.Input(
            shape=self.input_daily_prec_size,
            name='input_daily')
        input_high_freq = layers.Input(
            shape=self.input_high_freq_prec_size,
            name='input_high_freq')
        input_attributes = layers.Input(
            shape=self.input_attributes_size,
            name='input_attributes')

        if not self.options.combined_transformer:
            # Transformer for daily precipitation
            x_daily = input_daily
            for _ in range(self.options.nb_transformer_blocks_daily):
                x_daily = self.transformer_block(
                    x_daily,
                    num_heads=self.options.num_heads_daily,
                    ff_dim=self.options.ff_dim_daily,
                    dropout_rate=self.options.dropout_rate_daily,
                    activation=self.options.activation)

            # Transformer for high-frequency precipitation
            x_high_freq = input_high_freq
            for _ in range(self.options.nb_transformer_blocks_high_freq):
                x_high_freq = self.transformer_block(
                    x_high_freq,
                    num_heads=self.options.num_heads_high_freq,
                    ff_dim=self.options.ff_dim_high_freq,
                    dropout_rate=self.options.dropout_rate_high_freq,
                    activation=self.options.activation)

            # Transformer for attributes
            x_attributes = input_attributes
            for _ in range(self.options.nb_transformer_blocks_attributes):
                x_attributes = self.transformer_block(
                    x_attributes,
                    num_heads=self.options.num_heads_attributes,
                    ff_dim=self.options.ff_dim_attributes,
                    dropout_rate=self.options.dropout_rate_attributes,
                    activation=self.options.activation)

            # Concatenate
            x = layers.Concatenate(axis=-1)([x_daily, x_high_freq, x_attributes])

        else:
            x = layers.Concatenate(axis=1)([input_daily, input_high_freq])
            # Broadcast static attributes across timesteps
            x = tf.expand_dims(input_attributes, axis=1) + x

            for _ in range(self.options.nb_transformer_blocks_combined):
                x = self.transformer_block(
                    x,
                    num_heads=self.options.num_heads_combined,
                    ff_dim=self.options.ff_dim_combined,
                    dropout_rate=self.options.dropout_rate_combined,
                    activation=self.options.activation)

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
        x_ffn = layers.Dense(ff_dim, activation=activation)(x_attn)
        x_ffn = layers.Dropout(dropout_rate)(x_ffn)
        x_ffn = layers.Dense(ff_dim, activation=activation)(x_ffn)
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
