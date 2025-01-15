import torch
import torch.nn as nn


class CholeskyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, normalize=False):
        super(CholeskyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.normalize = normalize

        if self.normalize:
            self.input_norm = nn.LayerNorm(input_size)
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if self.normalize:
            x = self.input_norm(x)

        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]  # Take the last time step's hidden state

        if self.normalize:
            last_hidden_state = self.layer_norm(last_hidden_state)

        output = self.fc(last_hidden_state)
        return output




#other implementation
class CustomCholeskyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_assets, num_layers=1):
        super(CholeskyLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_assets = n_assets

        self.cells = nn.ModuleList(
            [
                LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        self.output_dim = (n_assets * (n_assets + 1)) // 2  # Lower triangular elements
        self.projection = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, init_states=None):
        batch_size, seq_length, _ = x.size()
        if init_states is None:
            init_states = [
                (
                    torch.zeros(batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(batch_size, self.hidden_dim).to(x.device),
                )
                for _ in range(self.num_layers)
            ]

        outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, *init_states[layer_idx])
                init_states[layer_idx] = (h_t, c_t)
                x_t = h_t  # Pass hidden state to next layer
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)  # Collect outputs for all timesteps
        final_output = self.projection(outputs[:, -1, :])  # Only last timestep
        return final_output


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_g = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_f.bias.data.fill_(1.0)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat((x, h_prev), dim=1)
        i_t = torch.sigmoid(self.W_i(combined))
        f_t = torch.sigmoid(self.W_f(combined))
        g_t = torch.tanh(self.W_g(combined))
        o_t = torch.sigmoid(self.W_o(combined))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
