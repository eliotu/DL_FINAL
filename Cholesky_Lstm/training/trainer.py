from Cholesky_Lstm.data_loading.dataset import CholeskyDataset
import torch
from torch.utils.data import DataLoader


def prepare_data(data, sequence_length, test_ratio=0.2):
    """
    Prepares input data for CholeskyLSTM

    Inputs:
        - sequence_length : integer
            NUmber of LSTM cells we wanna use (i.e. the lookback length)
        - test_ratio: float
            the ratio of the test set length compared full length of data (float between 0 and 1)
            test data will be formed as the last test_ratio percentage of the data

    Outputs:
        - train_dataset, test_dataset : Torch datasets ready for training and testing
    """

    full_dataset = CholeskyDataset(data, sequence_length)

    num_test = int(len(full_dataset) * test_ratio)
    num_train = len(full_dataset) - num_test

    train_dataset = torch.utils.data.Subset(full_dataset, list(range(num_train)))
    test_dataset = torch.utils.data.Subset(
        full_dataset, list(range(num_train, len(full_dataset)))
    )

    return train_dataset, test_dataset


def train_model(model, train_loader, criterion, optimizer, num_epochs, crit_dict=None):
    if crit_dict is not None:
        cr_values = {}
        for key, cr in crit_dict.items():
            cr_values[key] = []

    for epoch in range(num_epochs):
        model.train()

        all_predictions = []
        all_targets = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)

            all_predictions.append(outputs)
            all_targets.append(y_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        total_loss = criterion(all_predictions, all_targets)

        if crit_dict is not None:
            for key, cr in crit_dict.items():
                cr_values[key].append(cr(all_predictions, all_targets))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.9f}")

    return cr_values


def test_model(model, test_loader, criterion):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs)
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    total_loss = criterion(all_predictions, all_targets)

    return total_loss, all_predictions, all_targets
