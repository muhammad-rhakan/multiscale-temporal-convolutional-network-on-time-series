import pandas as pd
import numpy as np

from scipy import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Clean column names and date reformatting
def clean_data(
        data, 
        timestamp_col='Timestamp', 
        format='%d/%m/%Y %I:%M:%S %p', 
        target_col='Normal/Attack'):
    
    df = data.copy()

    # Reformat time for convenience
    df[timestamp_col] = df[timestamp_col].str.strip()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=format)

    # Encode target feature
    df[target_col] = (df[target_col]
                      .str.replace(" ","", regex=False)
                      .map({'Normal':0, 'Attack':1}))

    return df


# Fill missing timestamps from the sequence
def fill_missing_timestamps(
        data, actuators, sensors, target,
        timestamp_col='Timestamp', freq='s'):
    
    df = data.copy()

    # Get the missing timestamps within the range of min and max
    full_range = pd.date_range(
        start=df[timestamp_col].min(),
        end=df[timestamp_col].max(),
        freq=freq)

    missing_timestamps = full_range[~full_range.isin(df[timestamp_col])]
    print(f"Missing timestamp: {len(missing_timestamps)}")
    print(f"Dataset length: {len(df)}")

    df_filled = df.set_index(timestamp_col).reindex(full_range).rename_axis(timestamp_col).reset_index()

    df_filled[actuators] = df_filled[actuators].fillna(0)
    df_filled[sensors] = df_filled[sensors].interpolate(method='linear', limit_direction="both")
    df_filled[target] = df_filled[target].fillna(0)

    print(f"Dataset length after filling: {len(df_filled)}")
    print(f"Missing value remaining: {df_filled.isna().sum().sum()}")

    return  df_filled


# Filter out the initial operating state of the system
def filter_operating_state(data, target, hours):
    df = data.copy()

    samples_to_drop = 60 * 60 * hours
    df_filtered = df.loc[samples_to_drop:].reset_index(drop=True)

    print(f"Initial Length: {df.shape[0]}")
    print(f"Dataset after removed:  {df_filtered.shape[0]}")

    return df_filtered


# Apply normalization and PCA
class NormalPCAPreprocessor:
    def __init__(self, scaler, pca):
        self.scaler = scaler
        self.pca = pca

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)


# Main function to execute the preprocessing steps
def main():
    source_path = "datasets/raw/"
    output_path = "datasets/preprocessed/"
    val_ratio = 0.2
    pca_components = 16

    # ------------------------------
    # 1. Load raw dataset
    # -------------------------------
    normal = pd.read_csv(source_path + "SWaT_Dataset_Normal_v1.csv", sep=';')
    attack = pd.read_csv(source_path + "SWaT_Dataset_Attack_v0.csv", sep=';')

    normal.columns = normal.columns.str.strip()
    attack.columns = attack.columns.str.strip()

    timestamp = 'Timestamp'
    target = 'Normal/Attack'
    actuators = [a for a in normal.columns if (normal[a].nunique() <=3) and (a not in [timestamp, target])]
    sensors = [s for s in normal.columns if (s not in actuators) and (s not in [timestamp, target])]
    dataset_features = normal.drop(columns={timestamp, target}).columns

    # ----------------------------
    # 2. Clean and filter dataset
    # ----------------------------
    """ Normal Dataset """
    normal[dataset_features] = normal.loc[:,dataset_features].astype(str).replace(",", ".", regex=True)
    normal[dataset_features] = normal[dataset_features].apply(pd.to_numeric, errors='coerce')

    normal_clean = clean_data(normal)
    normal_fill = fill_missing_timestamps(normal_clean, actuators, sensors, target)
    normal_filtered = filter_operating_state(normal_fill, target, hours=4)

    """ Attack Dataset """
    attack[dataset_features] = attack.loc[:,dataset_features].astype(str).replace(",", ".", regex=True)
    attack[dataset_features] = attack[dataset_features].apply(pd.to_numeric, errors='coerce')

    attack_clean = clean_data(attack)
    attack_fill = fill_missing_timestamps(attack_clean, actuators, sensors, target)

    # ----------------
    # 3. Split dataset 
    # ----------------
    X_train = normal_filtered[dataset_features].values
    y_train = normal_filtered[target].values

    # Split attack dataset into validation and test sets
    X_attack = attack_fill[dataset_features].values
    y_attack = attack_fill[target].values

    val_size = int(len(X_attack) * val_ratio)
    X_val, X_test = X_attack[:val_size], X_attack[val_size:]
    y_val, y_test = y_attack[:val_size], y_attack[val_size:]

    
    # --------------------------
    # 4. Normalization and PCA
    # --------------------------
    scaler = StandardScaler()
    pca = PCA(n_components=pca_components)

    # Instantiate and execute
    preprocessor = NormalPCAPreprocessor(scaler, pca)
    preprocessor.fit(X_train, y_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)


    #----------------------------
    # Save preprocessed datasets
    #----------------------------
    np.savez_compressed(
        output_path + "preprocessed_datasets.npz",
        X_train=X_train_transformed,
        X_val=X_val_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test)
    print("Preprocessed datasets saved successfully.")


if __name__ == "__main__":
    main()