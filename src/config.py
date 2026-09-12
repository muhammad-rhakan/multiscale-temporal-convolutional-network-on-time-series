from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Premodeling setup
WINDOW_SHAPE = 120
TRAIN_STRIDE = 120
TEST_STRIDE = 10

# Framework setup
BATCH_SIZE = 256
LATENT_DIM = 10
EPOCHS = 50
LEARNING_RATE = 0.0005

def configuration_settings():
    checkpoint_path = "model/checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.weights.h5"
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        verbose=1)

    stoppage = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True)

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        mode='min',
        verbose=1)

    return [checkpoint, stoppage, lr_scheduler]