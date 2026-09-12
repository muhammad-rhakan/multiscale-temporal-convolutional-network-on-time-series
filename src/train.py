import numpy as np
import tensorflow as tf
from src.config import *
from model.mstcn import MultiScaleAutoEncoder


def train_model(train_ds, val_ds):
    TIME_STEPS = train_ds.shape[1]
    FEATURES = train_ds.shape[2]
    CONFIGS = configuration_settings()

    # Traininng setup
    train_tensor = tf.data.Dataset.from_tensor_slices((train_ds, train_ds))
    train_tensor = (
        train_tensor
        .shuffle(buffer_size=len(train_ds))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))
    
    # Validation setup
    val_tensor = tf.data.Dataset.from_tensor_slices((val_ds, val_ds))
    val_tensor = (
        val_tensor
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    model = MultiScaleAutoEncoder(features=FEATURES, latent_dim=LATENT_DIM)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

    history = model.fit(
        x=train_tensor,
        validation_data=val_tensor,
        epochs=EPOCHS,
        callbacks=CONFIGS,
        verbose=1)

    return model, history