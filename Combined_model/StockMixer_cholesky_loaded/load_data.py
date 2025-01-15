import sys

from Cholesky_Lstm.data_loading.data_loader import CustomCovDataLoader
from Cholesky_Lstm.utils.covariance import calculate_hourly_realized_covariance
from Combined_model.StockMixer_cholesky_loaded.utils.stock_dataloader import (
    StockDataLoader,
)

sys.path.append("../../")
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


def load_cholesky_data(path: str, n_first_stocks: int = 30) -> pd.DataFrame:
    data_loader = CustomCovDataLoader(path, n_first=n_first_stocks)
    merged_df = data_loader.load_data()
    merged_df_minutely = data_loader.transform_data(merged_df).copy()
    return calculate_hourly_realized_covariance(
        merged_df_minutely, freq="h", transform=False
    ), [name.split("_")[0] for name in merged_df_minutely.columns.tolist()]


def load_stock_data(
    path: str, tickers: list[str], data_normalization: str = None
) -> pd.DataFrame:
    loader = StockDataLoader(path)
    return loader.load_stocks(tickers, data_normalization=data_normalization)


class CholeskyStockDataset(Dataset):
    def __init__(
        self,
        minutely_data_zip_path: str,
        minutely_data_folder_path: str,
        sequence_length: int,
        prediction_horizon: int,
        n_first_stocks: int = 30,
        data_normalization: str = None,
        batch_normalization: str = None,
    ):
        self.cholesky_df, self.tickers = load_cholesky_data(
            minutely_data_zip_path, n_first_stocks
        )
        self.stock_dfs = load_stock_data(
            minutely_data_folder_path,
            self.tickers,
            data_normalization=data_normalization,
        )

        self.cholesky_data = self.cholesky_df.to_numpy()
        self.stock_data = np.array(
            [self.stock_dfs[ticker].to_numpy() for ticker in self.tickers]
        )

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.n_stocks = len(self.tickers)

        self.batch_normalization = batch_normalization

    def __len__(self):
        return len(self.cholesky_df) - self.sequence_length - self.prediction_horizon

    def __getitem__(self, idx):

        eod_data_batch = self.stock_data[:, idx : idx + self.sequence_length, :4]
        if self.batch_normalization == "z-score":
            eod_data_batch = (
                eod_data_batch - np.mean(eod_data_batch, axis=1, keepdims=True)
            ) / np.std(eod_data_batch, axis=1, keepdims=True)
        cholesky_vectors_batch = self.cholesky_data[idx : idx + self.sequence_length, :]
        mask_batch = np.min(
            self.stock_data[
                :, idx : idx + self.sequence_length + self.prediction_horizon, 4
            ],
            axis=1,
        )
        base_batch = self.stock_data[
            :, idx + self.sequence_length + self.prediction_horizon - 1, 5
        ]
        ground_truth_batch = self.stock_data[
            :, idx + self.sequence_length + self.prediction_horizon - 1, 6
        ]

        return (
            eod_data_batch,
            cholesky_vectors_batch,
            mask_batch,
            base_batch,
            ground_truth_batch,
        )


def create_dataloaders(
    dataset,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 1,
):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_indices = np.sort(indices[:train_size])
    valid_indices = np.sort(indices[train_size : train_size + valid_size])
    test_indices = np.sort(indices[train_size + valid_size :])

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    valid_loader = DataLoader(
        torch.utils.data.Subset(dataset, valid_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    zip_path = "../../data/Data_Cleaned.zip"
    data_path = "../../data/Data_60min_cleaned"
    data_normalization = None

    cholesky_df, tickers = load_cholesky_data(
        zip_path,
        n_first_stocks=30,
    )
    print(f"Tickers: {tickers}")
    print(f"Shape: {cholesky_df.shape}")
    print(cholesky_df.head())
    print(cholesky_df.tail())

    stock_data = load_stock_data(
        data_path, tickers, data_normalization=data_normalization
    )
    for ticker, df in stock_data.items():
        print(df.tail())
        print(f"\nSummary for {ticker}:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

    dataset = CholeskyStockDataset(
        zip_path,
        data_path,
        sequence_length=7,
        prediction_horizon=1,
        n_first_stocks=30,
        data_normalization=data_normalization,
    )

    print(f"Length of dataset:                  {len(dataset)}")
    item = dataset.__getitem__(0)
    print(f"Shape of eod_data_batch:            {item[0].shape}")
    print(f"Shape of cholesky_vectors_batch:    {item[1].shape}")
    print(f"Shape of mask_batch:                {item[2].shape}")
    print(f"Shape of base_batch:                {item[3].shape}")
    print(f"Shape of ground_truth_batch:        {item[4].shape}")
