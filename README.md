# eT2QFNN
This repository contains the MATLAB implementation of the evolving Type-2 Quantum Fuzzy Neural Network (eT2QFNN).
It uses interval type-2 quantum fuzzy sets with uncertain jump positions and trains fully online, automatically
growing and updating rules via a decoupled extended Kalman filter (DEKF).

## Requirements
- MATLAB with the Statistics and Machine Learning Toolbox (for `fitgmdist`).

## Data format
The input data matrix must be structured as:

```
data = [X, Y]
```

- `X` is the input feature matrix with `I` columns.
- `Y` is the target matrix with `M` columns.

The example script loads [databj.mat](databj.mat), which is expected to contain a `data` variable in that format.

## Quick start
Open and run the example script:

1. Edit [run_eT2QFNN.m](run_eT2QFNN.m) to set the number of inputs and learning rate.
2. Run the script in MATLAB.

Key parameters in the script:

- `I`: number of input features.
- `lr`: learning rate used in the DEKF update.
- `mode`: task type
	- `1`: binary classification
	- `2`: multiclass classification
	- `3`: regression
- `training_data`: number of samples used for online training before validation.

## Python example (Box-Jenkins)
This repo includes a Python example that downloads the Box-Jenkins airline passenger dataset
and runs eT2QFNN as a simple time-series regression task.

Run:

```
python run_et2qfnn_boxjenkins.py
```

The script downloads the dataset from:

```
https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

It converts the series to a supervised format using lagged inputs and runs
the regression mode (mode 3).

## Core function
The main entry point is:

```
[y, rule, network, performance] = eT2QFNN(data, I, lr, mode, training_data);
```

Where:

- `data`: full dataset, `N x (I + M)`.
- `I`: number of input features.
- `lr`: learning rate.
- `mode`: 1 (binary), 2 (multiclass), 3 (regression).
- `training_data`: number of samples used for online training.

Outputs:

- `y`: predicted outputs (class labels for classification, regression values for regression).
- `rule`: rule evolution details (`K`, `nrule`, `wr`).
- `network`: learned model parameters (centers, widths, weights, and quantum intervals).
- `performance`: metrics and timing depending on the mode.

## Notes
- The model grows rules online until `training_data` is reached.
- When `mode` is 3, the function reports RMSE and NDEI for training, validation, and total data.
- When `mode` is 1 or 2, the function reports classification rates for training and validation.

## Citation
If you use this code in your work, please cite the following reference and acknowledge the corresponding author:

```
@Inbook{et2qfnn,
author="Ashfahani, Andri
and Pratama, Mahardhika
and Lughofer, Edwin
and Cai, Qing
and Sheng, Huang",
editor="Lughofer, Edwin
and Sayed-Mouchaweh, Moamar",
title="An Online RFID Localization in the Manufacturing Shopfloor",
bookTitle="Predictive Maintenance in Dynamic Systems: Advanced Methods, Decision Support Tools and Real-World Applications",
year="2019",
publisher="Springer International Publishing",
pages="287--309",
isbn="978-3-030-05645-2",
doi="10.1007/978-3-030-05645-2_10",
}
```
