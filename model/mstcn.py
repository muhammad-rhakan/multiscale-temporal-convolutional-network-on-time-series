import numpy as np
import tensorflow as tf
import keras
import os

from keras.layers import (
    Layer,
    Conv1D, Conv1DTranspose,
    Activation, Dropout, LayerNormalization,
    Concatenate)
from keras.models import Model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides INFO and WARNING logs



class Backbone(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.gelu1 = Activation('gelu')
        self.gelu2 = Activation('gelu')
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # First block
        x = self.conv1(inputs)
        x = self.gelu1(x)
        x = self.dropout1(x, training=training)
        # Second block
        x = self.conv2(x)
        x = self.gelu2(x)
        return self.dropout2(x, training=training)


class MSTCN_EncoderBlock(Layer):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.size1 = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')

        self.size3_1 = Backbone(filters=32, kernel_size=3, dilation_rate=1)
        self.size3_2 = Backbone(filters=64, kernel_size=3, dilation_rate=2)

        self.size5_1 = Backbone(filters=32, kernel_size=5, dilation_rate=1)
        self.size5_2 = Backbone(filters=64, kernel_size=5, dilation_rate=2)

        self.concat_branches = Concatenate(axis=-1)
        self.compress = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')

    def call(self, inputs):
        # Feature extraction size 1
        s1 = self.size1(inputs)

        # Feature extraction size 3
        s3 = self.size3_1(inputs)
        s3 = self.size3_2(s3)

        # Feature extraction size 5
        s5 = self.size5_1(inputs)
        s5 = self.size5_2(s5)

        # Concatenate the outputs of the three branches and compress
        concat = self.concat_branches([s1, s3, s5])
        return self.compress(concat)


class TCNResidualBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, activation='relu', dropout=0.2, **kwargs):
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