import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Model Architecture (Unchanged) ---
class TrajectoryPredictor(nn.Module):
    """An encoder-decoder GRU model for UAV trajectory prediction."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(TrajectoryPredictor, self).__init__()
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.decoder_gru = nn.GRU(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_input, future_len):
        _, encoder_hidden = self.encoder_gru(x_input)
        decoder_input = x_input[:, -1, :].unsqueeze(1)
        decoder_hidden = encoder_hidden
        outputs = []
        for _ in range(future_len):
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            prediction = self.fc(decoder_output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            decoder_input = prediction.unsqueeze(1)
        return torch.cat(outputs, dim=1)

# --- 2. Helper Functions (Unchanged) ---
def denormalize_data(data, stats, use_whitening):
    """De-normalizes trajectory data back to its original scale."""
    if use_whitening:
        mean, l_matrix = stats['mean'], stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        # Reshape for matrix multiplication if necessary
        data_reshaped = data.reshape(-1, data.shape[-1])
        return (data_reshaped @ l_inv.T) + mean
    else:
        max_norm = stats.get('max_length', stats.get('max_magnitude', 1.0))
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    """Integrates velocity predictions to get a position trajectory."""
    displacements = np.cumsum(np.asarray(velocities) * dt, axis=0)
    return np.asarray(start_pos) + displacements

# --- 3. Main Inference and Visualization Block ---
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = 'best_model_4.pth'
    STATS_PATH = 'vel_stats_4.npz'
    VAL_SEGMENTS_PATH = 'val_segments_5.npz'
    
    # NEW: Choose which sub-trajectory to visualize
    SUB_TRAJECTORY_ID_TO_VIZ = 0

    # Model parameters (must match the saved model)
    USE_VELOCITY_PREDICTION = True
    USE_WHITENING = False 
    INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM = 3, 128, 3, 3
    DROPOUT_PROB, DT = 0.5, 0.1
    INPUT_SEQ_LEN, OUTPUT_SEQ_LEN = 20, 10

    # --- Load Model and Data ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrajectoryPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")

        stats = np.load(STATS_PATH)
        val_data = np.load(VAL_SEGMENTS_PATH)
        print(f"✅ Statistics and validation data loaded.")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e.filename}. Please check your paths.")
        exit()
    
    # --- Find all segments for the chosen sub-trajectory ---
    sub_traj_ids = val_data['sub_trajectory_ids']
    segment_indices = np.where(sub_traj_ids == SUB_TRAJECTORY_ID_TO_VIZ)[0]

    if len(segment_indices) == 0:
        print(f"❌ Error: No segments found for Sub-Trajectory ID #{SUB_TRAJECTORY_ID_TO_VIZ}.")
        print(f"   Available IDs range from {np.min(sub_traj_ids)} to {np.max(sub_traj_ids)}.")
        exit()
        
    print(f"Found {len(segment_indices)} segments for Sub-Trajectory ID #{SUB_TRAJECTORY_ID_TO_VIZ}.")

    # --- Reconstruct the full ground truth path ---
    # Get all the normalized input segments for our target sub-trajectory
    target_inputs_norm = val_data['input_segments'][:, :, segment_indices].transpose(2, 1, 0)
    
    # Stitch the segments together to form one long velocity sequence
    # We take the first full segment, then just the last point from all subsequent segments
    first_segment_vel = target_inputs_norm[0]
    subsequent_last_vels = target_inputs_norm[1:, -1, :]
    full_gt_vel_norm = np.vstack([first_segment_vel, subsequent_last_vels])
    
    # De-normalize and integrate to get the final position path
    full_gt_vel_denorm = denormalize_data(full_gt_vel_norm, stats, USE_WHITENING)
    full_true_pos_path = integrate_velocity(np.zeros(3), full_gt_vel_denorm, DT)
    
    # --- Generate all predictions along the path ---
    all_predicted_paths = []
    with torch.no_grad():
        for i, segment_idx in enumerate(segment_indices):
            # Get the normalized input data for this specific segment
            input_segment_norm = val_data['input_segments'][:, :, segment_idx].T
            
            # Prepare tensor for the model
            input_tensor = torch.from_numpy(input_segment_norm).float().unsqueeze(0).to(device)
            
            # Run prediction
            predicted_output_norm = model(input_tensor, future_len=OUTPUT_SEQ_LEN)
            predicted_output_norm_np = predicted_output_norm.cpu().squeeze(0).numpy()
            
            # De-normalize the predicted velocities
            predicted_output_denorm = denormalize_data(predicted_output_norm_np, stats, USE_WHITENING)
            
            # Find the starting position for this prediction on the true path
            # The prediction starts after the input sequence ends.
            start_pos_index = i + INPUT_SEQ_LEN
            if start_pos_index >= len(full_true_pos_path): continue # Avoid index out of bounds
            
            start_position = full_true_pos_path[start_pos_index - 1]
            
            # Integrate the predicted velocities to get a position path
            predicted_pos_path = integrate_velocity(start_position, predicted_output_denorm, DT)
            all_predicted_paths.append(predicted_pos_path)

    # --- Visualize on one graph ---
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the full, continuous ground truth path
    ax.plot(full_true_pos_path[:, 0], full_true_pos_path[:, 1], full_true_pos_path[:, 2], 'b-', linewidth=3, label='Ground Truth Path')

    # Plot all the predictions branching off from the path
    for i, path in enumerate(all_predicted_paths):
        label = 'Predictions' if i == 0 else "" # Only label the first one
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-o', markersize=2, alpha=0.8, label=label)

    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_zlabel('Z Coordinate (meters)')
    ax.set_title(f'All Predictions for Sub-Trajectory #{SUB_TRAJECTORY_ID_TO_VIZ}')
    ax.legend()
    ax.grid(True)
    
    print(f"✅ Plot generated. Showing graph for Sub-Trajectory #{SUB_TRAJECTORY_ID_TO_VIZ}.")
    plt.show()