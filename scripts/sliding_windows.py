import pandas as pd
import numpy as np

# Create sliding windows function
def sliding_windows(X, window_shape, stride):
    n_windows =  len(X) - window_shape + 1
    start = np.arange(0, n_windows, stride)
    windows = np.array([X[j:j+ window_shape] for j in start])
    return windows


# Apply sliding windows to the for train, validation, and test datasets
def create_window_datasets(X_train, X_val, X_test, y_train, y_val, y_test, window_shape, train_stride, test_stride):
    X_train_windows = sliding_windows(X_train, window_shape, train_stride)
    X_val_windows = sliding_windows(X_val, window_shape, test_stride)
    X_test_windows = sliding_windows(X_test, window_shape, test_stride)

    y_train_windows = sliding_windows(y_train, window_shape, train_stride)
    y_val_windows = sliding_windows(y_val, window_shape, test_stride)
    y_test_windows = sliding_windows(y_test, window_shape, test_stride)

    print(f"Training dataset shape: {X_train_windows.shape}")
    print(f"Validation dataset shape: {X_val_windows.shape}")
    print(f"Testing dataset shape: {X_test_windows.shape}")
    print(f"Validation labels: {y_val_windows.shape}")
    print(f"Ground Truth shape: {y_test_windows.shape}")

    return {
        "train_ds": X_train_windows,
        "val_ds": X_val_windows,
        "test_ds": X_test_windows,
        "train_labels": y_train_windows,
        "val_labels": y_val_windows,
        "test_labels": y_test_windows}


# Apply sliding windows to the datasets and save them
if __name__ == "__main__":
    source_path = "datasets/preprocessed/datasets.npz"
    output_path = "datasets/tensors/"

    # Load and extract preprocessed datasets
    X_train = np.load(source_path)["X_train"]
    X_val = np.load(source_path)["X_val"]
    X_test = np.load(source_path)["X_test"]
    y_train = np.load(source_path)["y_train"]
    y_val = np.load(source_path)["y_val"]
    y_test = np.load(source_path)["y_test"]

    datasets = create_window_datasets(X_train, X_val, X_test, y_train, y_val, y_test, window_shape=120, train_stride=10, test_stride=120)

    # Save datasets
    np.savez_compressed(
        output_path + "tensors.npz",
        train_ds=datasets['train_ds'],
        val_ds=datasets['val_ds'],
        test_ds=datasets['test_ds'],
        train_labels=datasets['train_labels'],
        val_labels=datasets['val_labels'],
        test_labels=datasets['test_labels'])

    print("Tensor datasets are saved successfully.")