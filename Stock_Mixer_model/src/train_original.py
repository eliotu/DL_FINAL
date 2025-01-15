import random
import numpy as np
import os
import torch as torch
from load_data import load_60min_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random


def main():

    np.random.seed(123456789)
    torch.random.manual_seed(12345678)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(device)

    stock_num = 30
    lookback_length = 16
    epochs = 100
    valid_index = 10558
    test_index = 11965
    fea_num = 4
    market_num = 20
    steps = 1
    learning_rate = 0.001
    alpha = 0.1
    scale_factor = 3

    logs = {
        "epoch": [],
        "train_loss": [],
        "train_reg_loss": [],
        "train_rank_loss": [],
        "valid_loss": [],
        "valid_reg_loss": [],
        "valid_rank_loss": [],
        "test_loss": [],
        "test_reg_loss": [],
        "test_rank_loss": [],
        "valid_mse": [],
        "valid_ic": [],
        "valid_ric": [],
        "valid_prec10": [],
        "valid_sr": [],
        "test_mse": [],
        "test_ic": [],
        "test_ric": [],
        "test_prec10": [],
        "test_sr": [],
    }

    dataset_path = "/Users/eliotullmo/Documents/ETHZ/COURSES/semester 3/DL/DL_final/data/Data_60min_cleaned"
    tickers = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIA",
        "DIS",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
    ]

    eod_data, mask_data, gt_data, price_data = load_60min_data(
        dataset_path, tickers, steps
    )

    print(
        "eod_data - Type:",
        type(eod_data),
        "| Dimensions:",
        getattr(eod_data, "shape", "N/A"),
        "| Data type:",
        getattr(eod_data, "dtype", "N/A"),
    )
    print(
        "mask_data - Type:",
        type(mask_data),
        "| Dimensions:",
        getattr(mask_data, "shape", "N/A"),
        "| Data type:",
        getattr(mask_data, "dtype", "N/A"),
    )
    print(
        "gt_data - Type:",
        type(gt_data),
        "| Dimensions:",
        getattr(gt_data, "shape", "N/A"),
        "| Data type:",
        getattr(gt_data, "dtype", "N/A"),
    )
    print(
        "price_data - Type:",
        type(price_data),
        "| Dimensions:",
        getattr(price_data, "shape", "N/A"),
        "| Data type:",
        getattr(price_data, "dtype", "N/A"),
    )

    feature_1 = eod_data[:, :, 0]

    windows = [4, 8, 16, 32]

    moving_averages = []

    for window in windows:
        ma = np.array(
            [
                np.convolve(stock, np.ones(window) / window, mode="valid")
                for stock in feature_1
            ]
        )

        padded_ma = np.pad(
            ma, ((0, 0), (window - 1, 0)), mode="constant", constant_values=np.nan
        )

        moving_averages.append(padded_ma)

    moving_avg_features = np.stack(moving_averages, axis=2)
    eod_data_expanded = np.concatenate((eod_data, moving_avg_features), axis=2)

    fea_num2 = fea_num + len(windows)

    print(
        "eod_data_expanded - Type:",
        type(eod_data_expanded),
        "| Dimensions:",
        eod_data_expanded.shape,
        "| Data type:",
        eod_data_expanded.dtype,
    )

    max_feature_1_per_stock = np.nanmax(eod_data[:, :, 0], axis=1, keepdims=True)

    eod_data_normalized = eod_data_expanded / max_feature_1_per_stock[:, :, np.newaxis]

    print(
        "eod_data_normalized - Type:",
        type(eod_data_normalized),
        "| Dimensions:",
        eod_data_normalized.shape,
        "| Data type:",
        eod_data_normalized.dtype,
    )
    print(
        "Maximum value after normalization for each stock (should be <= 1):",
        np.nanmax(eod_data_normalized[:, :, 0], axis=1),
    )

    print(
        "eod_data - Type:",
        type(eod_data),
        "| Dimensions:",
        getattr(eod_data, "shape", "N/A"),
        "| Data type:",
        getattr(eod_data, "dtype", "N/A"),
    )

    eod_data = eod_data_normalized[:, :, [0, 5, 6, 7]].astype(np.float32)

    random_stock = random.randint(0, stock_num - 1)
    print(f"📊 Zufällig ausgewählter Stock für EOD-Daten: {random_stock}")

    time_steps = np.arange(mask_data.shape[1])

    stock_eod_data = eod_data[random_stock, :, :]

    fig = go.Figure()

    for feature_idx in range(fea_num):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=stock_eod_data[:, feature_idx],
                mode="lines",
                name=f"Feature {feature_idx + 1}",
            )
        )

    fig.update_layout(
        title=f"📊 EOD-Daten für Stock {random_stock}",
        xaxis=dict(title="Zeitschritte"),
        yaxis=dict(title="Wert"),
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.5)",
        ),
    )

    fig.show()

    i = 0
    eod_data = eod_data[:, 6 + i :, :]
    mask_data = mask_data[:, 6 + i :]
    gt_data = gt_data[:, 6 + i :]
    price_data = price_data[:, 6 + i :]

    print(
        "eod_data - Type:",
        type(eod_data),
        "| Dimensions:",
        getattr(eod_data, "shape", "N/A"),
        "| Data type:",
        getattr(eod_data, "dtype", "N/A"),
    )
    print(
        "mask_data - Type:",
        type(mask_data),
        "| Dimensions:",
        getattr(mask_data, "shape", "N/A"),
        "| Data type:",
        getattr(mask_data, "dtype", "N/A"),
    )
    print(
        "gt_data - Type:",
        type(gt_data),
        "| Dimensions:",
        getattr(gt_data, "shape", "N/A"),
        "| Data type:",
        getattr(gt_data, "dtype", "N/A"),
    )
    print(
        "price_data - Type:",
        type(price_data),
        "| Dimensions:",
        getattr(price_data, "shape", "N/A"),
        "| Data type:",
        getattr(price_data, "dtype", "N/A"),
    )

    trade_dates = mask_data.shape[1]
    model = StockMixer(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        market=market_num,
        scale=scale_factor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def validate(start_index, end_index):
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            loss = 0.0
            reg_loss = 0.0
            rank_loss = 0.0
            for cur_offset in range(
                start_index - lookback_length - steps + 1,
                end_index - lookback_length - steps + 1,
            ):
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
                )
                prediction = model(data_batch)
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                    prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
                )
                loss += cur_loss.item()
                reg_loss += cur_reg_loss.item()
                rank_loss += cur_rank_loss.item()
                cur_valid_pred[
                    :, cur_offset - (start_index - lookback_length - steps + 1)
                ] = cur_rr[:, 0].cpu()
                cur_valid_gt[
                    :, cur_offset - (start_index - lookback_length - steps + 1)
                ] = gt_batch[:, 0].cpu()
                cur_valid_mask[
                    :, cur_offset - (start_index - lookback_length - steps + 1)
                ] = mask_batch[:, 0].cpu()
            loss = loss / (end_index - start_index)
            reg_loss = reg_loss / (end_index - start_index)
            rank_loss = rank_loss / (end_index - start_index)
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        return loss, reg_loss, rank_loss, cur_valid_perf

    def get_batch(offset=None):
        if offset is None:
            offset = random.randrange(0, valid_index)
        seq_len = lookback_length
        mask_batch = mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)

        return (
            eod_data[:, offset : offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1),
        )

    patience = 10
    tolerance = 1e-4
    early_stopping_counter = 0
    best_epoch = 0

    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None

    for epoch in range(epochs):
        print(
            f"Epoch {epoch + 1} ##########################################################"
        )
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0

        for j in range(valid_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
            )

            optimizer.zero_grad()
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
                prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
            )
            cur_loss.backward()
            optimizer.step()

            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()

        tra_loss /= valid_index - lookback_length - steps + 1
        tra_reg_loss /= valid_index - lookback_length - steps + 1
        tra_rank_loss /= valid_index - lookback_length - steps + 1

        print(
            f"Train : loss: {tra_loss:.2e}  =  {tra_reg_loss:.2e} + alpha*{tra_rank_loss:.2e}"
        )

        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(
            valid_index, test_index
        )
        print(
            f"Valid : loss: {val_loss:.2e}  =  {val_reg_loss:.2e} + alpha*{val_rank_loss:.2e}"
        )

        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(
            test_index, trade_dates
        )
        print(
            f"Test : loss: {test_loss:.2e}  =  {test_reg_loss:.2e} + alpha*{test_rank_loss:.2e}"
        )

        if val_loss < best_valid_loss - tolerance:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
            early_stopping_counter = 0
            best_epoch = epoch + 1
            print(
                f"Validation loss improved to {best_valid_loss:.2e}. Resetting patience."
            )
        else:
            early_stopping_counter += 1
            print(
                f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        print(
            "Valid performance:\n",
            f"mse: {val_perf['mse']:.2e}, IC: {val_perf['IC']:.2e}, RIC: {val_perf['RIC']:.2e}, "
            f"prec@10: {val_perf['prec_10']:.2e}, SR: {val_perf['sharpe5']:.2e}",
        )
        print(
            "Test performance:\n",
            f"mse: {test_perf['mse']:.2e}, IC: {test_perf['IC']:.2e}, RIC: {test_perf['RIC']:.2e}, "
            f"prec@10: {test_perf['prec_10']:.2e}, SR: {test_perf['sharpe5']:.2e}\n",
        )

        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_valid_loss:.2e} (Epoch {best_epoch})"
            )
            break

        logs["epoch"].append(epoch + 1)
        logs["train_loss"].append(tra_loss)
        logs["train_reg_loss"].append(tra_reg_loss)
        logs["train_rank_loss"].append(tra_rank_loss)
        logs["valid_loss"].append(val_loss)
        logs["valid_reg_loss"].append(val_reg_loss)
        logs["valid_rank_loss"].append(val_rank_loss)
        logs["test_loss"].append(test_loss)
        logs["test_reg_loss"].append(test_reg_loss)
        logs["test_rank_loss"].append(test_rank_loss)
        logs["valid_mse"].append(val_perf["mse"])
        logs["valid_ic"].append(val_perf["IC"])
        logs["valid_ric"].append(val_perf["RIC"])
        logs["valid_prec10"].append(val_perf["prec_10"])
        logs["valid_sr"].append(val_perf["sharpe5"])
        logs["test_mse"].append(test_perf["mse"])
        logs["test_ic"].append(test_perf["IC"])
        logs["test_ric"].append(test_perf["RIC"])
        logs["test_prec10"].append(test_perf["prec_10"])
        logs["test_sr"].append(test_perf["sharpe5"])

    def plot_logs(logs):
        epochs = logs["epoch"]

        # Losses
        plt.figure()
        plt.plot(epochs, logs["train_loss"], label="Train Loss")
        plt.plot(epochs, logs["valid_loss"], label="Valid Loss")
        plt.plot(epochs, logs["test_loss"], label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()
        plt.show()

        # IC and RIC
        plt.figure()
        plt.plot(epochs, logs["valid_ic"], label="Valid IC")
        plt.plot(epochs, logs["test_ic"], label="Test IC")
        plt.plot(epochs, logs["valid_ric"], label="Valid RIC")
        plt.plot(epochs, logs["test_ric"], label="Test RIC")
        plt.xlabel("Epochs")
        plt.ylabel("IC / RIC")
        plt.title("Information Coefficient and Rank IC")
        plt.legend()
        plt.show()

        # Precision@10 and Sharpe Ratio
        plt.figure()
        plt.plot(epochs, logs["valid_prec10"], label="Valid Prec@10")
        plt.plot(epochs, logs["test_prec10"], label="Test Prec@10")
        plt.plot(epochs, logs["valid_sr"], label="Valid Sharpe Ratio")
        plt.plot(epochs, logs["test_sr"], label="Test Sharpe Ratio")
        plt.xlabel("Epochs")
        plt.ylabel("Precision@10 / Sharpe Ratio")
        plt.title("Precision@10 and Sharpe Ratio")
        plt.legend()
        plt.show()

    plot_logs(logs)

    with open("training_logs.pkl", "wb") as f:
        pickle.dump(logs, f)


if __name__ == "__main__":
    main()
