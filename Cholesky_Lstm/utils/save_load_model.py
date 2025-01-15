import os
import torch


def save_cholesky_lstm(model, save_path):
    """
    Save a CholeskyLSTM model.

    Args:
        model: The CholeskyLSTM model to save
        save_path (str): Path where to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({"model_state_dict": model.state_dict()}, save_path)

    print(f"Model saved to {save_path}")


def load_cholesky_lstm(model_path, model):
    """
    Load a saved CholeskyLSTM model.

    Args:
        model_path (str): Path to the saved model file
        model: the model class

    Returns:
        model: The loaded CholeskyLSTM model
    """
    model_info = torch.load(model_path)

    # Load the state dict
    model.load_state_dict(model_info["model_state_dict"])
    return model
