import sys

from Cholesky_Lstm.models.lstm import CholeskyLSTM
from Stock_Mixer_model.src.model import MultTime2dMixer, NoGraphMixer
sys.path.append("../../")
import torch
import torch.nn as nn


class CholeskyStockMixer(nn.Module):
    def __init__(
        self,
        stocks,
        time_steps,
        channels,
        market,
        scale_dim,
        cholesky_vector_size,
        cholesky_hidden_size,
        lstm_model_path = '../../src/LSTM/model/cholesky_lstm_model_eliot.pth',
        dropout_rate: int = None
    ):
        super(CholeskyStockMixer, self).__init__()

        self.stocks = stocks
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # # Cholesky LSTM
        model_info = torch.load(lstm_model_path)
        print(model_info["hyperparameters"])
        # print(f"Loading trained CholeskyLSTM with hyperparameters:")
        # print(f"Hidden size:    {model_info['hyperparameters']['hidden_size']}")
        # self.cholesky_lstm = cholesky_rnn.CholeskyLSTM(
        #     input_size=cholesky_vector_size,
        #     hidden_size=cholesky_hidden_size,
        #     output_size=cholesky_vector_size
        # ).to(self.device).double()  # Convert to float64

        # Cholesky LSTM
        print(f"Loading trained CholeskyLSTM with hyperparameters:")
        print(f"Input size:     {model_info['hyperparameters']['input_size']}")
        print(f"Hidden size:    {model_info['hyperparameters']['hidden_size']}")
        print(f"N_assets:       {model_info['hyperparameters']['n_assets']}")
        self.cholesky_lstm = CholeskyLSTM(
            input_dim=model_info['hyperparameters']['input_size'],
            hidden_dim=model_info['hyperparameters']['hidden_size'],
            n_assets=model_info['hyperparameters']['n_assets']
        ).to(self.device).double()  # Convert to float64

        # Load the model parameters
        self.cholesky_lstm.load_state_dict(model_info['model_state_dict'])

        for param in self.cholesky_lstm.parameters():
            param.requires_grad = False  # Freeze weights of the Cholesky LSTM
        
        # Stock Mixer with updated dimensions
        combined_features = channels + cholesky_vector_size
        self.mixer = MultTime2dMixer(
            time_steps,
            combined_features,
            scale_dim=scale_dim
        ).to(self.device).double()  # Convert to float64
        
        
        self.channel_fc = nn.Linear(combined_features, 1, dtype=torch.float64).to(self.device)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1, dtype=torch.float64).to(self.device)
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            dtype=torch.float64
        ).to(self.device)
        self.stock_mixer = NoGraphMixer(stocks, market).to(self.device).double()  # Convert to float64
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1, dtype=torch.float64).to(self.device)

    def reconstruct_covariance(self, cholesky_vec):
        # Convert vectorized Cholesky back to matrix form
        n = self.stocks
        L = torch.zeros(cholesky_vec.shape[0], n, n, dtype=torch.float64).to(self.device)
        idx = torch.tril_indices(n, n)
        L[:, idx[0], idx[1]] = cholesky_vec
        # Compute covariance matrix: L @ L.T
        cov = torch.bmm(L, L.transpose(1, 2))
        return cov

    def forward(self, price_data, cholesky_data):
        # Ensure input data is float64
        price_data = price_data.double()
        cholesky_data = cholesky_data.double()
        
        # Get Cholesky prediction
        cholesky_output = self.cholesky_lstm(cholesky_data)

        price_data = price_data.squeeze(0)
        x = price_data.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        
        expanded_cholesky = cholesky_output.expand(x.shape[0], -1)
        expanded_cholesky = expanded_cholesky.unsqueeze(1)
        expanded_cholesky = expanded_cholesky.expand(-1, x.shape[1], -1)
        x = torch.cat([x, expanded_cholesky], dim=-1)

        # Expand price_data with cholesky vectors
        expanded_cholesky_for_price = cholesky_output.expand(price_data.shape[0], -1)
        expanded_cholesky_for_price = expanded_cholesky_for_price.unsqueeze(1)
        expanded_cholesky_for_price = expanded_cholesky_for_price.expand(-1, price_data.shape[1], -1)
        price_data = torch.cat([price_data, expanded_cholesky_for_price], dim=-1)

        # Process through the model
        y = self.mixer(price_data, x)
        y = self.channel_fc(y).squeeze(-1)
        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)

        return y + z