from Cholesky_Lstm.data_loading.data_loader import CustomCovDataLoader
import torch
from torch.utils.data import DataLoader
from Cholesky_Lstm.models.lstm import CholeskyLSTM, CustomCholeskyLSTM
from Cholesky_Lstm.training.trainer import prepare_data, test_model, train_model
from Cholesky_Lstm.utils.covariance import calculate_hourly_realized_covariance
from Cholesky_Lstm.utils.loss import (
    FrobeniusLoss,
    FrobeniusLossWithSignPenalty,
    SignAccuracyLoss,
)
from Cholesky_Lstm.utils.save_load_model import save_cholesky_lstm


def main():
    zip_path = "/Users/eliotullmo/Documents/ETHZ/COURSES/semester 3/DL/DL_final/data/Data_Cleaned.zip"

    DfLoader = CustomCovDataLoader(zip_path, n_first=5)
    merged_df = DfLoader.load_data()
    merged_df_minutely = DfLoader.transform_data(merged_df).copy()

    cholesky_vectors = calculate_hourly_realized_covariance(
        merged_df_minutely, freq="h", return_cholesky=True, transform=False
    )
    print(f"Shape of Cholesky vectors: {cholesky_vectors.shape}")

    sequence_length = 7
    train_dataset, test_dataset = prepare_data(cholesky_vectors, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    print(
        f"Length of train loader: {len(train_loader)*128}. Test: {len(test_loader)*128}"
    )

    input_size = cholesky_vectors.shape[1]  # Length of Cholesky vector
    hidden_size = 465  # Can be tuned
    output_size = cholesky_vectors.shape[1]
    n_assets = int(
        (-1 + (1 + 8 * input_size) ** 0.5) / 2
    )  # Calculate number of assets from length of Cholesky vector

    model = CholeskyLSTM(input_size, hidden_size, output_size, normalize=False)
    # choose the two models

    # model = CustomCholeskyLSTM(input_size, hidden_size, n_assets)

    criterion = FrobeniusLossWithSignPenalty(lambda_penalty=5e-07)  ## winning 5e-07
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)  ## winning 1e-05

    criterion1 = FrobeniusLossWithSignPenalty(lambda_penalty=5e-07)
    criterion2 = FrobeniusLoss()
    criterion3 = SignAccuracyLoss()

    crit_dict = {"f_norm": criterion1, "f_norm_pen": criterion2, "acc": criterion3}

    cr_values = train_model(
        model, train_loader, criterion, optimizer, num_epochs=100, crit_dict=crit_dict
    )

    save_cholesky_lstm(model, "models/cholesky_lstm.pth")

    print("TESTING")

    test_loss, predictions_test, targets_test = test_model(
        model, test_loader, criterion
    )

    criterion1 = FrobeniusLossWithSignPenalty(lambda_penalty=5e-7)
    criterion2 = FrobeniusLoss()
    criterion3 = SignAccuracyLoss()

    f_loss_pen = criterion1(predictions_test, targets_test)
    f_loss = criterion2(predictions_test, targets_test)
    acc = criterion3(predictions_test, targets_test)

    print(f"criterion1: {1- acc}, criterion2: {f_loss}, criterion3: {f_loss_pen}")


if __name__ == "__main__":
    main()
