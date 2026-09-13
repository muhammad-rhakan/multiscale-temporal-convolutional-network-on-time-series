# Multi-Scale Temporal Convolutional Network Autoencoder

Autoencoder architecture for time-series anomaly detection with a multi-scale encoder block and multiple TCN blocks on the decoder layer.



## 🛠️ Architecture Design
![architecture design](/assets/architecture.png)

## Dataset
The SWaT dataset used for this model could not be disclosed due to confidentiality. However, the dataset can be  accessed upon request via:
```
https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/
```

The raw data consists of two xlsx files:

|File|Description|
|-----|-----|
|SWaT_Dataset_Normal_v1| 4 days normal operation for model training |
|SWaT_Dataset_Attack_v0| 7 days attacked operations for validation and testing |

## 📚 References
* J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A Dataset to Support Research in the Design of Secure Water Treatment Systems.”
* X. Ma, F. Wu, J. Yue, P. Feng, X. Peng, and J. Chu, “MSE-TCN: Multi-scale temporal convolutional network with channel attention for open-set gas classification,” Microchemical Journal, vol. 207, Dec. 2024.

## Project Structure
```
Multi-Scale-TCN-Autoencoder/
├── assets/                        # Resources
├── model/
│   └── mstcn.py                   # Model architecture
├── notebooks/
│   └── data_analysis.ipynb        # Exploratory data analysis
├── src/
│   ├── config.py                  # Predefined hyperparameters and model configurations
│   ├── detect.py                  # Anomaly detection from reconstructed errors
│   ├── evaluate.py                # Evaluation metrics
│   ├── preprocessing.py           # Data preprocessing steps
│   ├── sliding_window.py          # Sliding window for neural network input
│   └── train.py                   # Model training
├── .gitignore
├── main.py                        # Code execution from training to evaluation
└── README.md
```
