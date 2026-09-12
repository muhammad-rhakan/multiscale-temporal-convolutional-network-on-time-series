import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# Use the Mean Squared Error to calculate the anomaly scores
def calculate_anomaly_scores(mse, labels, eval_points=True):
    """
    Parameters:
        - mse: Mean Squared Error values
        - labels: Ground truth labels
        - eval_points: Whether to evaluate individual points or aggregated values (window-wise)
        
    Returns:
        - If eval_points is True, returns flattened MSE and labels
        - If eval_points is False, returns mean MSE and max labels per sample

    """
    if eval_points:
        return mse.flatten(), labels.flatten()
    
    else:
        return np.mean(mse, axis=1), np.max(labels, axis=1)


# Use static thresholding to determine the threshold for anomaly detection
def static_thresholding(anomaly_scores, static_type, n_percentile, k):
    """
    Parameters:
        - anomaly_scores: Anomaly scores calculated from MSE
        - static_type: Type of static thresholding ('percentile' or 'std_dev')
        - n_percentile: Percentile value for thresholding (if static_type is 'percentile')
        - k: Number of standard deviations for thresholding (if static_type is 'std_dev')

    """
    if static_type == "percentile":
        return np.percentile(anomaly_scores, n_percentile)
    else:
        return np.mean(anomaly_scores) + k * np.std(anomaly_scores)



# Use the threshold to flag anomalies based on the anomaly scores
def flag_anomalies(
        train_mse,
        test_mse,
        test_labels,
        eval_points,
        static_type,
        n_percentile=None,
        parameter_k=None):
    """
    Parameters:
        - train_mse: Mean Squared Error values for the training set as lookup distribution
        - test_mse: Mean Squared Error values for the test set
        - test_labels: Ground truth labels for the test set
        - eval_points: Whether to evaluate individual points or aggregated values (window-wise)
        - static_type: Type of static thresholding ('percentile' or 'std_dev')
        - n_percentile: Percentile value for thresholding (if static_type is 'percentile')
        - parameter_k: Number of standard deviations for thresholding (if static_type is 'std_dev')

    Returns:
        - Dictionary containing the threshold, precision, recall, and F1 score of the anomaly detection
    """

    # Determine a threshold based on the lookup distribution and the specified static thresholding method
    lookup_distribution = np.mean(train_mse, axis=1)
    threshold = static_thresholding(lookup_distribution, static_type, n_percentile, parameter_k)

    # Flag anomalies based on the threshold and calculate precision, recall, and F1 score
    anomaly_scores, ground_truth = calculate_anomaly_scores(test_mse, test_labels, eval_points)
    flags = (anomaly_scores > threshold).astype(int)

    return {
        "Parameter": f"{n_percentile}%" if static_type == "percentile" else f"{parameter_k} std",
        "Threshold": round(float(threshold), 4),
        "Precision": round(precision_score(ground_truth, flags, zero_division=0), 4),
        "Recall": round(recall_score(ground_truth, flags, zero_division=0), 4),
        "F1 Score": round(f1_score(ground_truth, flags, zero_division=0), 4)}
