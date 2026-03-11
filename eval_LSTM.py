import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# --- 1. Model Architecture ---
# This definition remains unchanged. It must match your trained model.
class TrajectoryPredictorLSTM(nn.Module):
    """
    An encoder-decoder LSTM model for UAV trajectory prediction.
    This architecture is a modification of the provided GRU-based model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        """
        Initializes the TrajectoryPredictorLSTM model.

        Args:
            input_dim (int): The number of features in the input data (e.g., 3 for x, y, z).
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            output_dim (int): The number of features in the output data (e.g., 3 for x, y, z).
            num_layers (int): The number of recurrent layers in the LSTM.
            dropout_prob (float): Dropout probability between LSTM layers.
        """
        super(TrajectoryPredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_input, future_len, y_target=None, teacher_forcing=False):
        """
        Forward pass of the model.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_seq_len, input_dim).
            future_len (int): Length of the future trajectory to predict.
            y_target (torch.Tensor, optional): Ground truth future trajectory 
                of shape (batch_size, future_len, output_dim), required if teacher_forcing is True.
            teacher_forcing (bool): If True, uses ground truth as decoder input during training.

        Returns:
            torch.Tensor: Predicted trajectory of shape (batch_size, future_len, output_dim).
        """
        batch_size = x_input.size(0)

        # --- Encoder Pass ---
        # The LSTM returns output, (hidden_state, cell_state)
        # We only need the final hidden and cell states to initialize the decoder.
        _, (encoder_hidden, encoder_cell) = self.encoder_lstm(x_input)
        # encoder_hidden shape: (num_layers, batch_size, hidden_dim)
        # encoder_cell shape: (num_layers, batch_size, hidden_dim)

        # --- Decoder Initialization ---
        # The decoder's first input is the last observed coordinate from the input sequence.
        decoder_input = x_input[:, -1, :].unsqueeze(1)  # shape (batch_size, 1, input_dim)
        # The decoder's initial hidden state is the encoder's final hidden state tuple.
        decoder_hidden = (encoder_hidden, encoder_cell)

        outputs = []

        # --- Decoding Loop ---
        for t in range(future_len):
            # The decoder_hidden is a tuple (hidden_state, cell_state)
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            
            # Pass the decoder output through the fully connected layer to get a prediction.
            prediction = self.fc(decoder_output.squeeze(1))  # (batch_size, output_dim)

            outputs.append(prediction.unsqueeze(1))  # (batch_size, 1, output_dim)

            # --- Decide Next Decoder Input ---
            if teacher_forcing and y_target is not None:
                # Use the ground truth as the next input
                decoder_input = y_target[:, t, :].unsqueeze(1)
            else:
                # Use the current prediction as the next input
                decoder_input = prediction.unsqueeze(1)

        # Concatenate predictions along the time axis
        outputs = torch.cat(outputs, dim=1)  # (batch_size, future_len, output_dim)

        return outputs
# --- 2. Helper Functions ---

def denormalize_data(data, stats, use_whitening):
    """De-normalizes trajectory data back to its original scale."""
    if data.ndim == 3: # Handle batch dimension
        return np.array([denormalize_data(sample, stats, use_whitening) for sample in data])
        
    if use_whitening:
        mean = stats['mean']
        l_matrix = stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data @ l_inv.T) + mean
    else: # Max L2-Norm
        max_norm = stats.get('max_magnitude') or stats.get('max_length')
        if max_norm is None:
            raise KeyError("Could not find 'max_magnitude' or 'max_length' in the stats file.")
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    """Integrates velocity predictions to get a position trajectory."""
    displacements = np.cumsum(velocities * dt, axis=0)
    return start_pos + displacements

# --- 3. Metric Calculation ---

def calculate_metrics(model, dataloader, stats, config, device):
    """
    Calculates ADE, FDE, and MSE/RMSE for the entire validation set.
    """
    model.eval()
    all_ades = []
    all_fdes = []
    all_mses = [] # New list to store squared errors
    
    with torch.no_grad():
        for batch_inputs, batch_truths in dataloader:
            batch_inputs = batch_inputs.to(device)
            future_len = batch_truths.shape[1]
            
            # Get model predictions
            batch_predictions = model(batch_inputs, future_len)

            # Convert to numpy arrays for calculation
            inputs_np = batch_inputs.cpu().numpy()
            truths_np = batch_truths.cpu().numpy()
            preds_np = batch_predictions.cpu().numpy()
            
            # De-normalize all data
            inputs_denorm = denormalize_data(inputs_np, stats, config['USE_WHITENING'])
            truths_denorm = denormalize_data(truths_np, stats, config['USE_WHITENING'])
            preds_denorm = denormalize_data(preds_np, stats, config['USE_WHITENING'])
            
            # Iterate over each trajectory in the batch
            for i in range(len(inputs_denorm)):
                if config['USE_VELOCITY_PREDICTION']:
                    # Integrate velocities to get position trajectories
                    input_pos = integrate_velocity(np.zeros(3), inputs_denorm[i], config['DT'])
                    last_known_pos = input_pos[-1]
                    true_future_pos = integrate_velocity(last_known_pos, truths_denorm[i], config['DT'])
                    pred_future_pos = integrate_velocity(last_known_pos, preds_denorm[i], config['DT'])
                else:
                    # Data is already in position format
                    true_future_pos = truths_denorm[i]
                    pred_future_pos = preds_denorm[i]
                
                # Calculate L2 distance (Euclidean error) at each time step
                errors = np.linalg.norm(pred_future_pos - true_future_pos, axis=1)
                
                # NEW: Calculate squared errors for MSE
                squared_errors = np.square(errors)
                
                # Append metrics for this trajectory
                all_ades.append(np.mean(errors))
                all_fdes.append(errors[-1])
                all_mses.append(np.mean(squared_errors))




    # Calculate the final average metrics over all trajectories
    mean_ade = np.mean(all_ades)
    mean_fde = np.mean(all_fdes)
    mean_mse = np.mean(all_mses)
    
    return mean_ade, mean_fde, mean_mse

def measure_inference_speed(model, dataloader, device, num_batches=10):
    """
    Measures average inference time per batch over a limited number of batches.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for i, (batch_inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            batch_inputs = batch_inputs.to(device)
            future_len = 10  # or use fixed value or output.shape[1]

            start = time.time()
            _ = model(batch_inputs, future_len)
            end = time.time()

            times.append(end - start)

    avg_batch_time = np.mean(times)
    print(f"🕒 Average inference time per batch: {avg_batch_time:.6f} seconds")
    print(f"⚡ Approximate FPS: {len(batch_inputs) / avg_batch_time:.2f} samples/second")


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    # !! IMPORTANT: Update these paths and settings to match your experiment !!
    config = {
        'MODEL_PATH': 'best_model_LSTM.pth',
        'STATS_PATH': 'vel_stats_4.npz',
        'VAL_SEGMENTS_PATH': 'val_segments_4.npz',
        'USE_VELOCITY_PREDICTION': True,
        'USE_WHITENING': False,
        'INPUT_DIM': 3,
        'HIDDEN_DIM': 128,
        'NUM_LAYERS': 3,
        'OUTPUT_DIM': 3,
        'DROPOUT_PROB': 0.5,
        'DT': 0.1,
        'BATCH_SIZE': 64 # You can adjust this based on your GPU memory
    }

    # --- Load Model and Data ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrajectoryPredictorLSTM(
            config['INPUT_DIM'], config['HIDDEN_DIM'], config['OUTPUT_DIM'],
            config['NUM_LAYERS'], config['DROPOUT_PROB']
        )
        model.load_state_dict(torch.load(config['MODEL_PATH'], map_location=device))
        model.to(device)
        print(f"✅ Model loaded successfully from '{config['MODEL_PATH']}' and moved to {device}.")

        stats = np.load(config['STATS_PATH'])
        print(f"✅ Statistics loaded from '{config['STATS_PATH']}'.")
        
        val_data = np.load(config['VAL_SEGMENTS_PATH'])
        print(f"✅ Validation data loaded from '{config['VAL_SEGMENTS_PATH']}'.")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e.filename}")
        print("Please update file paths in the 'config' dictionary.")
        exit()

    # --- Prepare DataLoader ---
    # The raw data shape is (features, seq_len, num_samples).
    # We permute it to (num_samples, seq_len, features) for the DataLoader.
    input_segments = np.transpose(val_data['input_segments'], (2, 1, 0))
    output_segments = np.transpose(val_data['output_segments'], (2, 1, 0))

    # Create PyTorch tensors
    input_tensor = torch.from_numpy(input_segments).float()
    output_tensor = torch.from_numpy(output_segments).float()

    # Create dataset and dataloader
    val_dataset = TensorDataset(input_tensor, output_tensor)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    print(f"✅ Created DataLoader with {len(val_dataset)} validation samples.")
    
    # --- Calculate and Report Metrics ---
    print("\nCalculating metrics over the entire validation set...")
    start_time = time.time()
    
    ade, fde, mse = calculate_metrics(model, val_loader, stats, config, device)
    rmse = np.sqrt(mse) # Calculate RMSE from MSE
    
    end_time = time.time()
    print(f"Calculation finished in {end_time - start_time:.2f} seconds.")

    # --- Measure Inference Speed ---
    print("\n⚡ Measuring inference speed over a few batches...")
    measure_inference_speed(model, val_loader, device, num_batches=10)

    print("\n--- Research Paper Metrics ---")
    print(f"📈 Average Displacement Error (ADE): {ade:.4f} meters")
    print(f"🎯 Final Displacement Error (FDE):   {fde:.4f} meters")
    print(f"🔲 Mean Squared Error (MSE):       {mse:.4f} meters²")
    print(f"📏 Root Mean Squared Error (RMSE): {rmse:.4f} meters")
    print("----------------------------\n")