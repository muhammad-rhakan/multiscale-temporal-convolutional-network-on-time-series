import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


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
def determine_threshold(anomaly_scores, static_type, n_percentile, k):
    """
    Parameters:
        - anomaly_scores: Anomaly scores calculated from MSE
        - static_type: Type of static thresholding ('percentile' or 'std_dev')
        - n_percentile: Percentile value for thresholding (if static_type is 'percentile')
        - k: Number of standard deviations for thresholding (if static_type is 'std_dev')
    """
    if static_type == "percentile":
        if n_percentile is None or k is not None:
            raise ValueError("Percentile threshold only requires argument n_percentile")
        else:
            return np.percentile(anomaly_scores, n_percentile)
        
    elif static_type.isin("stddev", "gaussian"):
        if n_percentile is not None or k is None:
            raise ValueError("Std_Dev threshold only requires argument k")
        else:
            return np.mean(anomaly_scores) + k * np.std(anomaly_scores)

    else:
        raise ValueError("Unknown static type")
    

# Use the threshold to flag anomalies based on the anomaly scores
def detect_anomalies(
    test_mse,
    test_labels,
    eval_points,
    threshold=None,
    reference_errors=None,
    static_type=None,
    n_percentile=None,
    parameter_k=None):
    """
    Parameters:
        - threshold: Boundary to classify normal/anomalous points.
                     If threshold hasnt been determined, calculate from baseline errors
        - baseline_errors: Mean Squared Error values set as lookup distribution, can be training or validation error set
        - test_mse: Mean Squared Error values for the test set
        - test_labels: Ground truth labels for the test set
        - eval_points: Whether to evaluate individual points or aggregated values (window-wise)
        - static_type: Type of static thresholding ('percentile' or 'std_dev')
        - n_percentile: Percentile value for thresholding (if static_type is 'percentile')
        - parameter_k: Number of standard deviations for thresholding (if static_type is 'std_dev')
    """

    if threshold is None:
        if reference_errors is None:
            raise ValueError("Baseline errors required if threshold is not yet determined.")
        if static_type not in ("percentile", "stddev", "gaussian"):
            raise ValueError("static_type must be 'percentile' or 'std_dev' when threshold is not provided.")

        lookup_distribution = np.mean(reference_errors, axis=1)
        threshold = determine_threshold(lookup_distribution, static_type, n_percentile, parameter_k)

    # Flag anomalies based on the threshold and calculate precision, recall, and F1 score
    anomaly_scores, ground_truth = calculate_anomaly_scores(test_mse, test_labels, eval_points)

    return (anomaly_scores > threshold).astype(int)
