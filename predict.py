import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Re-define the Model Architecture ---
# This should be the exact same architecture used for training.
class TrajectoryPredictor(nn.Module):
    """
    An encoder-decoder GRU model for UAV trajectory prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.decoder_gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_input, future_len):
        # x_input shape: (batch_size, seq_len, input_dim)
        batch_size = x_input.size(0)
        
        # Encoder
        _, encoder_hidden = self.encoder_gru(x_input)
        
        # The last known point from the input sequence will be the first input to the decoder
        decoder_input = x_input[:, -1, :].unsqueeze(1)
        decoder_hidden = encoder_hidden
        
        outputs = []

        # Decoder loop
        for _ in range(future_len):
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            prediction = self.fc(decoder_output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            
            # Use the current prediction as the input for the next time step
            decoder_input = prediction.unsqueeze(1)
            
        outputs = torch.cat(outputs, dim=1)
        return outputs

# --- 2. Helper Functions for Data Transformation ---

def denormalize_data(data, stats, use_whitening):
    """De-normalizes trajectory data back to its original scale."""
    if use_whitening:
        mean = stats['mean']
        l_matrix = stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        # Ensure data is a 2D array for matrix multiplication
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data @ l_inv.T) + mean
    else: # Max L2-Norm
        if 'max_magnitude' in stats:
            max_norm = stats['max_magnitude']
        elif 'max_length' in stats:
            max_norm = stats['max_length']
        else:
            raise KeyError("Could not find 'max_magnitude' or 'max_length' in the stats file.")
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    """Integrates velocity predictions to get a position trajectory."""
    # Ensure start_pos and velocities are numpy arrays
    start_pos = np.asarray(start_pos)
    velocities = np.asarray(velocities)
    
    # Calculate displacements by cumulative sum of (velocity * dt)
    displacements = np.cumsum(velocities * dt, axis=0)
    return start_pos + displacements

# --- 3. Main Inference and Visualization Block ---
if __name__ == '__main__':
    # --- Configuration ---
    #!! IMPORTANT: Update these paths to your files!!
    MODEL_PATH = 'best_model_1.pth'
    STATS_PATH = 'vel_stats_1.npz'
    VAL_SEGMENTS_PATH = 'val_segments.npz'
    SAMPLE_INDEX = 126175 # Which sample from the validation file to test

    # Match these settings to the model you are loading
    USE_VELOCITY_PREDICTION = True
    USE_WHITENING = False 

    # Model parameters (must match the saved model)
    INPUT_DIM = 3
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    OUTPUT_DIM = 3
    DROPOUT_PROB = 0.5
    DT = 0.1

    # --- Load Model and Statistics ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrajectoryPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")

        stats = np.load(STATS_PATH)
        print(f"Successfully loaded statistics from {STATS_PATH}")
        
        val_data = np.load(VAL_SEGMENTS_PATH)
        print(f"Successfully loaded validation data from {VAL_SEGMENTS_PATH}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        print("Please update MODEL_PATH, STATS_PATH, and VAL_SEGMENTS_PATH variables.")
        exit()

    # --- Load and Prepare Sample Data from val_segments.npz ---
    # CORRECTED: This is the critical fix.
    # The raw data shape is (features, seq_len, num_samples).

    # 1. Select the specific sample using SAMPLE_INDEX from the last axis.
    # This results in a shape of (features, seq_len), e.g., (3, 20).
    input_sample_raw = val_data['input_segments'][:, :, SAMPLE_INDEX]
    true_output_sample_raw = val_data['output_segments'][:, :, SAMPLE_INDEX]

    # 2. Transpose to get the desired (seq_len, features) shape, e.g., (20, 3).
    input_segment_normalized = input_sample_raw.T
    true_output_segment_normalized = true_output_sample_raw.T

    # --- Run Prediction ---
    with torch.no_grad():
        # 1. Prepare model input (already normalized)
        # Add a batch dimension to get shape (1, seq_len, features).
        input_tensor = torch.from_numpy(input_segment_normalized).float().unsqueeze(0).to(device)
        
        # 2. Perform inference
        # CORRECTED: Pass an integer for future_len, not a tuple.
        output_seq_len = true_output_segment_normalized.shape[0]
        predicted_output_normalized = model(input_tensor, future_len=output_seq_len)
        
        # 3. Convert prediction back to numpy array
        predicted_output_normalized_np = predicted_output_normalized.cpu().squeeze(0).numpy()
        
        # 4. De-normalize all segments for visualization
        input_history_denormalized = denormalize_data(input_segment_normalized, stats, USE_WHITENING)
        true_future_denormalized = denormalize_data(true_output_segment_normalized, stats, USE_WHITENING)
        predicted_future_denormalized = denormalize_data(predicted_output_normalized_np, stats, USE_WHITENING)

        # 5. Get final position trajectories for plotting
        if USE_VELOCITY_PREDICTION:
            # For visualization, integrate velocities to get a position path.
            # Assume a starting point of [0, 0, 0] for the history segment.
            input_history_pos = integrate_velocity(np.zeros(3), input_history_denormalized, DT)
            
            # Future paths start from the end of the history path.
            last_known_pos = input_history_pos[-1]
            true_future_pos = integrate_velocity(last_known_pos, true_future_denormalized, DT)
            predicted_trajectory_pos = integrate_velocity(last_known_pos, predicted_future_denormalized, DT)
        else:
            # If predicting positions directly
            input_history_pos = input_history_denormalized
            true_future_pos = true_future_denormalized
            predicted_trajectory_pos = predicted_future_denormalized

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the historical path
    ax.plot(input_history_pos[:, 0], input_history_pos[:, 1], input_history_pos[:, 2], 'b-', label='Input History')
    ax.scatter(input_history_pos[-1, 0], input_history_pos[-1, 1], input_history_pos[-1, 2], c='b', marker='o', s=60, label='Last Known Point')
    
    # Plot the ground truth future path
    ax.plot(true_future_pos[:, 0], true_future_pos[:, 1], true_future_pos[:, 2], 'g--', label='True Future')
    
    # Plot the predicted future path
    ax.plot(predicted_trajectory_pos[:, 0], predicted_trajectory_pos[:, 1], predicted_trajectory_pos[:, 2], 'r-o', markersize=4, label='Predicted Trajectory')

    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_zlabel('Z Coordinate (meters)')
    ax.set_title(f'Trajectory Prediction for Validation Sample #{SAMPLE_INDEX}')
    ax.legend()
    ax.grid(True)
    # Improve axis scaling for better visualization
    all_points = np.vstack([input_history_pos, true_future_pos, predicted_trajectory_pos])
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    plt.show()
