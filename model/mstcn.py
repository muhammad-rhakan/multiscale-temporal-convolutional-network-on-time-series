import numpy as np
import tensorflow as tf
import keras

from keras.layers import (
    Model, Layer,
    Conv1D, Conv1DTranspose,
    Activation, Dropout, LayerNormalization,
    Concatenate)


class Backbone(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2):
        super().__init__()
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.gelu = Activation('gelu')
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.gelu(x)
        return self.dropout(x)


class MSTCN_EncoderBlock(Layer):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.branch1_bb1 = Backbone(filters=32, kernel_size=3, dilation_rate=1)
        self.branch1_bb2 = Backbone(filters=64, kernel_size=3, dilation_rate=2)

        self.branch2_conv = Conv1D(filters=32, kernel_size=1, padding='same')

        self.branch3_bb1 = Backbone(filters=32, kernel_size=5, dilation_rate=1)
        self.branch3_bb2 = Backbone(filters=64, kernel_size=5, dilation_rate=2)

        self.concat_branches = Concatenate(axis=-1)
        self.compress = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')

    def call(self, inputs):
        b1 = self.branch1_bb1(inputs)
        b1 = self.branch1_bb2(b1)

        b2 = self.branch2_conv(inputs)

        b3 = self.branch3_bb1(inputs)
        b3 = self.branch3_bb2(b3)

        c = self.concat_branches([b1, b2, b3])
        return self.compress(c)


class TCNResidualBlock(tf.keras.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, decoder=True, activation='relu', dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv1 = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        self.norm1 = LayerNormalization()
        self.relu1 = Activation(activation)
        self.dropout1 = Dropout(dropout)

        self.conv2 = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        self.norm2 = LayerNormalization()
        self.relu2 = Activation(activation)
        self.dropout2 = Dropout(dropout)

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut = Conv1D(filters=self.filters, kernel_size=1, padding='same')
        else:
            self.shortcut = Activation('linear')
        super().build(input_shape)
    
    def call(self, X, training=None):
        h = self.conv1(X)
        h = self.norm1(h)
        h = self.relu1(h)
        h = self.dropout1(h, training=training)

        h = self.conv2(h)
        h = self.norm2(h)
        out = self.relu2(h + self.shortcut(X))

        return self.dropout2(out, training=training)


class MultiScaleAutoEncoder(Model):
    def __init__(self, features, latent_dim, dropout_rate=0.2):
        super().__init__()
        self.encoder_blocks = MSTCN_EncoderBlock()
        self.compress = Conv1D(filters=latent_dim, kernel_size=2, strides=2, activation='relu', padding='valid')
        self.expand = Conv1DTranspose(filters=48, kernel_size=2, strides=2, activation='relu', padding='valid')

        self.decoder3 = TCNResidualBlock(filters=64, kernel_size=2, dilation_rate=4)
        self.decoder2 = TCNResidualBlock(filters=64, kernel_size=2, dilation_rate=2)
        self.decoder1 = TCNResidualBlock(filters=64, kernel_size=2, dilation_rate=1)

        self.out = Conv1D(filters=features, kernel_size=1, activation='linear')

    def call(self, inputs):
        encoder = self.encoder_blocks(inputs)
        bottleneck = self.compress(encoder)

        d = self.expand(bottleneck)
        d = self.decoder3(d)
        d = self.decoder2(d)
        d = self.decoder1(d)

        return self.out(d)