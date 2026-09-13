from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)


def evaluate_detection_results(ground_truth, detection_result):
    cm = confusion_matrix(ground_truth, detection_result)

    TN, FP, FN, TP = cm.ravel()

    metrics = {
        "F1 Score": f1_score(ground_truth, detection_result),
        "Precision": precision_score(ground_truth, detection_result),
        "Recall": recall_score(ground_truth, detection_result),
        "Accuracy": accuracy_score(ground_truth, detection_result),
        "FPR": FP / (FP + TN),
        "FNR": FN / (FN + TP)
    }

    print("=== Model Performance Summary ===")
    for metric, value in metrics.items():
        print(f"{metric:<10}: {value:.4f}")