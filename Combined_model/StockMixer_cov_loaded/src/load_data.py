import os
import pandas as pd
import numpy as np


def load_covariance_matrices(base_path):
    """
    Loads the saved covariance matrices for training and testing from .npy files.

    Args:
        base_path (str): The path where the .npy files are stored.

    Returns:
        dict: A dictionary with four numpy arrays:
            - pred_train
            - trg_train
            - pred_test
            - trg_test
    """
    files = {
        "pred_train": "cov_matrices_pred_train.npy",
        "trg_train": "cov_matrices_trg_train.npy",
        "pred_test": "cov_matrices_pred_test.npy",
        "trg_test": "cov_matrices_trg_test.npy",
    }

    # Load Covariance Matrixes
    matrices = {}
    for key, filename in files.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            matrices[key] = np.load(file_path)
            print(f"{key} erfolgreich geladen. Form: {matrices[key].shape}")
        else:
            raise FileNotFoundError(
                f"Datei {filename} wurde im Pfad {base_path} nicht gefunden."
            )

    return matrices


def load_60min_data(data_path, tickers, steps=1):
    """
    Loads 60-minute data from CSV files, calculates EOD data, masks,
    ground truth returns and base prices. Trims the time series to the minimum length.

    Args:
        data_path (str): Path to the folder with CSV files.
        tickers (list): List of ticker symbols without file extension.
        steps (int): Step size for the calculation of returns.

    Returns:
        eod_data (np.array): Array with features (open, high, low, close).
        masks (np.array): Mask for missing values.
        ground_truth (np.array): Return as ground truth.
        base_price (np.array): Includes close prices for calculations.
    """
    min_length = float("inf")

    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}_60min_data_cleaned.csv")
        df = pd.read_csv(file_path)
        min_length = min(min_length, len(df))

    print(f"Minimale Länge der Zeitreihen: {min_length}")

    eod_data = []
    masks = []
    ground_truth = []
    base_price = []

    for idx, ticker in enumerate(tickers):
        file_path = os.path.join(data_path, f"{ticker}_60min_data_cleaned.csv")
        df = pd.read_csv(file_path)

        df = df[["open", "high", "low", "close"]][:min_length].values

        mask = ~np.isnan(df[:, -1])
        mask = mask.astype(np.float32)

        base = df[:, -1]  # 'close' Price
        ground_truth_vals = np.zeros_like(base)
        for i in range(steps, len(base)):
            ground_truth_vals[i] = (base[i] - base[i - steps]) / base[i - steps]

        if idx == 0:
            eod_data = np.zeros(
                (len(tickers), min_length, df.shape[1]), dtype=np.float32
            )
            masks = np.ones((len(tickers), min_length), dtype=np.float32)
            ground_truth = np.zeros((len(tickers), min_length), dtype=np.float32)
            base_price = np.zeros((len(tickers), min_length), dtype=np.float32)

        eod_data[idx, :, :] = df
        masks[idx, :] = mask
        ground_truth[idx, :] = ground_truth_vals
        base_price[idx, :] = base

    return eod_data, masks, ground_truth, base_price
