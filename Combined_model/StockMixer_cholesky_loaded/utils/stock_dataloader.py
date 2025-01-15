from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


class StockDataLoader:
    """Loads and processes stock data, maintaining individual DataFrames for each stock."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_stocks(self, tickers: List[str], steps: int = 1, data_normalization: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for multiple tickers, returning a dictionary of DataFrames.
        Each DataFrame contains OHLC data, masks, and calculated metrics.

        Args:
            tickers: List of stock ticker symbols
            steps: Number of periods for return calculation

        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")

        stock_data = {}

        for ticker in tickers:
            try:
                # Load single stock data
                df = self._load_single_stock(ticker, steps, data_normalization=data_normalization)
                stock_data[ticker] = df
                print(f"Successfully loaded {ticker} with {len(df)} rows")

            except Exception as e:
                print(f"Error loading {ticker}: {str(e)}")
                continue

        return stock_data

    def _load_single_stock(self, ticker: str, steps: int, data_normalization: str = None) -> pd.DataFrame:
        """Load and process data for a single stock."""
        # Load raw data
        file_path = self.data_path / f"{ticker}_60min_data_cleaned.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {ticker}")

        df = pd.read_csv(file_path)

        # Validate columns
        required_cols = {'timestamp', 'open', 'high', 'low', 'close'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

        # Process timestamps
        timestamps = pd.to_datetime(df['timestamp'])
        # df.set_index('timestamp', inplace=True)

        if data_normalization == "z-score":
            opens = (np.array(df['open'].values) - np.mean(df['open'].values)) / np.std(df['open'].values)
            highs = (np.array(df['high'].values) - np.mean(df['high'].values)) / np.std(df['high'].values)
            lows = (np.array(df['low'].values) - np.mean(df['low'].values)) / np.std(df['low'].values)
            closes = (np.array(df['close'].values) - np.mean(df['close'].values)) / np.std(df['close'].values)
        else:
            opens = np.array(df['open'].values)
            highs = np.array(df['high'].values)
            lows = np.array(df['low'].values)
            closes = np.array(df['close'].values)


        # Calculate masks for missing data
        masks = (~np.isnan(closes)).astype(np.float32)
        bases = np.array(df['close'].values).astype(np.float32)
        ground_truths = np.zeros_like(bases)
        for i in range(steps, len(bases)):
            ground_truths[i] = (bases[i] - bases[i - steps]) / bases[i - steps]

        df['open'] = opens
        df['high'] = highs
        df['low'] = lows
        df['close'] = closes
        df['mask'] = masks
        df['base'] = bases
        df['ground_truth'] = ground_truths
        
        # Set the timestamp as the index
        df.set_index(timestamps, inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'mask', 'base', 'ground_truth']]

    @staticmethod
    def _calculate_returns(prices: pd.Series, steps: int) -> pd.Series:
        """Calculate returns over specified number of steps."""
        returns = prices.pct_change(periods=steps)
        return returns

    def align_timestamps(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all stock DataFrames to have the same timestamp range.
        Useful when you need synchronized data across stocks.
        """
        if not stock_data:
            return {}

        # Find common date range
        common_dates = None
        for df in stock_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates &= set(df.index)

        common_dates = sorted(common_dates)

        # Align all dataframes
        aligned_data = {}
        for ticker, df in stock_data.items():
            aligned_data[ticker] = df.loc[common_dates]

        return aligned_data


if __name__ == "__main__":
    path = "../../../data/Data_60min_cleaned"
    tickers = ["AAPL", "AMGN", "AXP", "BA", "CAT"]
    loader = StockDataLoader(path)

    # Load individual stock data
    stock_data = loader.load_stocks(tickers, data_normalization="z-score")
    print(stock_data)
    # Access individual stock data
    aapl_data = stock_data["AAPL"]
    print(f"AAPL data shape: {aapl_data.shape}")

    # Get aligned data if needed
    aligned_data = loader.align_timestamps(stock_data)

    # Example analysis
    for ticker, df in stock_data.items():
        print(f"\nSummary for {ticker}:")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Average trading volume: {df['trading_volume'].mean():.4f}")
        print(f"Mean return: {df['returns'].mean():.4f}")
