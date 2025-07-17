import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    """
    An encoder-decoder GRU model for UAV trajectory prediction.
    This architecture is inspired by the VECTOR paper.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        """
        Initializes the TrajectoryPredictor model.

        Args:
            input_dim (int): The number of features in the input data (e.g., 3 for x, y, z).
            hidden_dim (int): The number of features in the hidden state of the GRU.
            output_dim (int): The number of features in the output data (e.g., 3 for x, y, z).
            num_layers (int): The number of recurrent layers in the GRU.
            dropout_prob (float): Dropout probability between GRU layers.
        """
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_input, future_len):
        """
        Forward pass of the model.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_seq_len, input_dim).
            future_len (int): Length of the future trajectory to predict.

        Returns:
            torch.Tensor: Predicted trajectory of shape (batch_size, future_len, output_dim).
        """
        batch_size = x_input.size(0)

        # Encoder pass
        _, encoder_hidden = self.encoder_gru(x_input)
        # encoder_hidden shape: (num_layers, batch_size, hidden_dim)

        # Decoder initialization
        decoder_input = x_input[:, -1, :].unsqueeze(1)  # shape (batch_size, 1, input_dim)
        decoder_hidden = encoder_hidden

        outputs = []

        for _ in range(future_len):
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            prediction = self.fc(decoder_output.squeeze(1))  # (batch_size, output_dim)

            outputs.append(prediction.unsqueeze(1))  # (batch_size, 1, output_dim)

            # For next step, use current prediction as decoder input
            decoder_input = prediction.unsqueeze(1)

        # Concatenate predictions along time axis
        outputs = torch.cat(outputs, dim=1)  # (batch_size, future_len, output_dim)

        return outputs
