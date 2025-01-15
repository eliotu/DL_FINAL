import random
import numpy as np
import os
import torch as torch

# from load_data import load_EOD_data
from load_data import load_60min_data, load_covariance_matrices
from evaluator import evaluate
from model import get_loss, StockMixer, StockMixer_new1, StockMixer_new2
import pickle
import pickle
import matplotlib.pyplot as plt


np.random.seed(123456789)
torch.random.manual_seed(12345678)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)

stock_num = 30
lookback_length = 16
epochs = 100
valid_index = 10558  # 14077 = number of total values
test_index = 11965  # 14077 = number of total values
fea_num = 4
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = "GELU"

# Initialize a dictionary for the logs
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
    "VZ",
    "V",
    "WBA",
    "WMT",
]
eod_data, mask_data, gt_data, price_data = load_60min_data(dataset_path, tickers, steps)

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


base_path = "Combined_model/StockMixer_cov_loaded/dataset/Covariance Matrix"
cov_matrices = load_covariance_matrices(base_path)
pred_train = cov_matrices["pred_train"]
trg_train = cov_matrices["trg_train"]
pred_test = cov_matrices["pred_test"]
trg_test = cov_matrices["trg_test"]
pred_cov = np.concatenate((pred_train, pred_test), axis=0)

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


trade_dates = mask_data.shape[1] - 1


model = StockMixer_new1(
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

            batch = get_batch(cur_offset)

            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), batch[:4]
            )

            cov_mat = torch.Tensor(batch[4]).to(device)
            prediction = model(data_batch, cov_mat)
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
        pred_cov[offset + seq_len + steps - 1, :, :],  # 30x30 Kovarianzmatrix
    )


patience = 40
patience_counter = 0
min_delta = 0.005e-5
for epoch in range(epochs):
    print(
        "epoch{}##########################################################".format(
            epoch + 1
        )
    )
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0

    for j in range(valid_index - lookback_length - steps + 1):
        batch = get_batch(batch_offsets[j])

        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device), batch[:4]
        )

        cov_mat = torch.Tensor(batch[4]).to(device)

        optimizer.zero_grad()
        prediction = model(data_batch, cov_mat)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
            prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
        )
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()

    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)

    print(
        "Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
            tra_loss, tra_reg_loss, tra_rank_loss
        )
    )

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print(
        "Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
            val_loss, val_reg_loss, val_rank_loss
        )
    )

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(
        test_index, trade_dates
    )
    print(
        "Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
            test_loss, test_reg_loss, test_rank_loss
        )
    )

    if val_loss < best_valid_loss - min_delta:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early Stopping Counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early Stopping aktiviert. Training wird beendet.")
            break

    print(
        "Valid performance:\n",
        "mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}".format(
            val_perf["mse"],
            val_perf["IC"],
            val_perf["RIC"],
            val_perf["prec_10"],
            val_perf["sharpe5"],
        ),
    )
    print(
        "Test performance:\n",
        "mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}".format(
            test_perf["mse"],
            test_perf["IC"],
            test_perf["RIC"],
            test_perf["prec_10"],
            test_perf["sharpe5"],
        ),
        "\n\n",
    )

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


# Show Plots
plot_logs(logs)


# Save Training Logs
with open("training_logs.pkl", "wb") as f:
    pickle.dump(logs, f)
