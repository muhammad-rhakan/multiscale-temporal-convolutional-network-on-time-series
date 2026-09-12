import numpy as np
from src.train import train_model
from src.detect import detect_anomalies
import argparse


parser = argparse.ArgumentParser(description="Parse anomaly detection parameters.")
parser.add_argument("--eval_points", action=argparse.BooleanOptionalAction, default=True, help="Whether to evaluate individual points (default: True)")
parser.add_argument("--static_type", type=str, choices=["percentile", "stddev"], default="percentile", help="Thresholding strategy to use: 'percentile' or 'stddev'",)
parser.add_argument("--n_percentile", type=float, default=None, help="Percentile cutoff when --static_type is 'percentile' (e.g., 95)")
parser.add_argument("--k", type=float, default=None, help="Multiplier for standard deviation when --static_type is 'stddev' (e.g., 3.0)")
args = parser.parse_args()

# Conditional validation & defaults based on static_type
kwargs = {"eval_points": args.eval_points, "static_type": args.static_type}

if args.static_type == "percentile":
    if args.k is not None:
        parser.error("Argument --k is not valid when --static_type is 'percentile'")
    kwargs["n_percentile"] = args.n_percentile if args.n_percentile is not None else 95.0

elif args.static_type == "stddev":
    if args.n_percentile is not None:
        parser.error("Argument --n_percentile is not valid when --static_type is 'stddev'")
    kwargs["k"] = args.k if args.k is not None else 3.0


def main():
    #-------------------
    # Load the datasets
    #-------------------
    with np.load("datasets/tensors/tensors.npz") as data:
        train_ds = data["train_ds"]
        val_ds = data["val_ds"]
        test_ds = data["test_ds"]

        test_labels = data["test_labels"]

    #-----------------
    # Train the model
    #-----------------
    model, history = train_model(train_ds, val_ds)

    #--------------------------------------
    # Make predictions and calculate errors
    #--------------------------------------
    predictions = model.predict(test_ds)
    errors = np.mean(np.square(test_ds - predictions), axis=2)

    #--------------------------
    # Perform anomaly detection
    #--------------------------
    # Get threshold from training error distribution
    train_predictions = model.predict(train_ds)
    train_errors = np.mean(np.square(train_ds - train_predictions), axis=2)

    result = detect_anomalies(
        train_errors,
        errors,
        test_labels,
        **kwargs)
    
    return result

if __name__ == "__main__":
    main()