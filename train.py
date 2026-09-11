from scripts.variables import FEATURES, LATENT_DIM, LEARNING_RATE, EPOCHS, BATCH_SIZE
from scripts.configs import configuration_settings
from model.mstcn import MultiScaleAutoEncoder

import tensorflow as tf
import numpy as np


def main():
    train_ds = np.load("datasets/tensors/tensors.npz")["train_ds"]
    val_ds = np.load("datasets/tensors/tensors.npz")["val_ds"]
    test_ds = np.load("datasets/tensors/tensors.npz")["test_ds"]
    val_labels = np.load("datasets/tensors/tensors.npz")["val_labels"]
    test_labels = np.load("datasets/tensors/tensors.npz")["test_labels"]
    
    TIME_STEPS = train_ds.shape[1]
    FEATURES = train_ds.shape[2]
    CONFIGS = configuration_settings()

    # Traininng setup
    train_tensor = tf.data.Dataset.from_tensor_slices((train_ds, train_ds))
    train_tensor = train_tensor.shuffle(buffer_size=len(train_ds)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Validation setup
    val_tensor = tf.data.Dataset.from_tensor_slices((val_ds, val_ds))
    val_tensor = val_tensor.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Testing setup
    test_tensor = tf.data.Dataset.from_tensor_slices((test_ds, test_ds))
    test_tensor = test_tensor.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    MSTCN_AE = MultiScaleAutoEncoder(features=FEATURES, latent_dim=LATENT_DIM)
    MSTCN_AE.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    MSTCN_AE.fit(
        data=train_tensor,
        validation_data=val_tensor,
        epochs=EPOCHS,
        callbacks=CONFIGS,
        verbose=1)


if __name__ == "__main__":
    main()