from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def configuration_settings():
    checkpoint_path = "checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.weights.h5"
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