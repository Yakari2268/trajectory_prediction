import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Define the Model Architecture ---
class TrajectoryPredictor(nn.Module):
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
        batch_size = x_input.size(0)
        _, encoder_hidden = self.encoder_gru(x_input)
        decoder_input = x_input[:, -1, :].unsqueeze(1)
        decoder_hidden = encoder_hidden
        outputs = []
        for _ in range(future_len):
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            prediction = self.fc(decoder_output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            decoder_input = prediction.unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

# --- 2. Helper Functions ---

def denormalize_data(data, stats, use_whitening):
    if use_whitening:
        mean = stats['mean']
        l_matrix = stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data @ l_inv.T) + mean
    else:
        if 'max_magnitude' in stats:
            max_norm = stats['max_magnitude']
        elif 'max_length' in stats:
            max_norm = stats['max_length']
        else:
            raise KeyError("Could not find 'max_magnitude' or 'max_length' in the stats file.")
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    start_pos = np.asarray(start_pos)
    velocities = np.asarray(velocities)
    displacements = np.cumsum(velocities * dt, axis=0)
    return start_pos + displacements

# --- 3. Main Script ---

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = 'best_model_6.pth'
    STATS_PATH = 'vel_stats_4.npz'
    VAL_SEGMENTS_PATH = 'val_segments_6.npz'

    USE_VELOCITY_PREDICTION = True
    USE_WHITENING = False

    INPUT_DIM = 3
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    OUTPUT_DIM = 3
    DROPOUT_PROB = 0.5
    DT = 0.1

    # --- Load Model and Data ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrajectoryPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")

        stats = np.load(STATS_PATH)
        print(f"Loaded statistics from {STATS_PATH}")

        val_data = np.load(VAL_SEGMENTS_PATH)
        print(f"Loaded validation data from {VAL_SEGMENTS_PATH}")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        exit()

    # --- Iterate and Compute ADE ---
    num_samples = val_data['input_segments'].shape[2]
    ade_list = []

    for sample_index in range(0,1000):
        input_sample_raw = val_data['input_segments'][:, :, sample_index]
        true_output_sample_raw = val_data['output_segments'][:, :, sample_index]
        
        input_segment_normalized = input_sample_raw.T
        true_output_segment_normalized = true_output_sample_raw.T

        with torch.no_grad():
            input_tensor = torch.from_numpy(input_segment_normalized).float().unsqueeze(0).to(device)
            output_seq_len = true_output_segment_normalized.shape[0]
            predicted_output_normalized = model(input_tensor, future_len=output_seq_len)
            predicted_output_normalized_np = predicted_output_normalized.cpu().squeeze(0).numpy()
            
            # De-normalize
            true_future_denormalized = denormalize_data(true_output_segment_normalized, stats, USE_WHITENING)
            predicted_future_denormalized = denormalize_data(predicted_output_normalized_np, stats, USE_WHITENING)

            if USE_VELOCITY_PREDICTION:
                input_history_denormalized = denormalize_data(input_segment_normalized, stats, USE_WHITENING)
                last_known_pos = integrate_velocity(np.zeros(3), input_history_denormalized, DT)[-1]
                true_future_pos = integrate_velocity(last_known_pos, true_future_denormalized, DT)
                predicted_future_pos = integrate_velocity(last_known_pos, predicted_future_denormalized, DT)
            else:
                true_future_pos = true_future_denormalized
                predicted_future_pos = predicted_future_denormalized

            errors = np.linalg.norm(true_future_pos - predicted_future_pos, axis=1)
            ade = errors.mean()
            ade_list.append((sample_index, ade))

    # Sort and print top 5
    ade_list.sort(key=lambda x: x[1])
    print("\nTop 5 sample indexes with least ADE:")
    for idx, error in ade_list[:5]:
        print(f"Sample Index: {idx}, ADE: {error:.4f}")

    # --- Visualization for a specific sample (optional, e.g., sample_index = ade_list[0][0]) ---
    SAMPLE_INDEX = ade_list[0][0]  # Visualize the best sample

    input_sample_raw = val_data['input_segments'][:, :, SAMPLE_INDEX]
    true_output_sample_raw = val_data['output_segments'][:, :, SAMPLE_INDEX]
    input_segment_normalized = input_sample_raw.T
    true_output_segment_normalized = true_output_sample_raw.T

    with torch.no_grad():
        input_tensor = torch.from_numpy(input_segment_normalized).float().unsqueeze(0).to(device)
        output_seq_len = true_output_segment_normalized.shape[0]
        predicted_output_normalized = model(input_tensor, future_len=output_seq_len)
        predicted_output_normalized_np = predicted_output_normalized.cpu().squeeze(0).numpy()

        input_history_denormalized = denormalize_data(input_segment_normalized, stats, USE_WHITENING)
        true_future_denormalized = denormalize_data(true_output_segment_normalized, stats, USE_WHITENING)
        predicted_future_denormalized = denormalize_data(predicted_output_normalized_np, stats, USE_WHITENING)

        if USE_VELOCITY_PREDICTION:
            input_history_pos = integrate_velocity(np.zeros(3), input_history_denormalized, DT)
            last_known_pos = input_history_pos[-1]
            true_future_pos = integrate_velocity(last_known_pos, true_future_denormalized, DT)
            predicted_future_pos = integrate_velocity(last_known_pos, predicted_future_denormalized, DT)
        else:
            input_history_pos = input_history_denormalized
            true_future_pos = true_future_denormalized
            predicted_future_pos = predicted_future_denormalized

    # --- Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(input_history_pos[:, 0], input_history_pos[:, 1], input_history_pos[:, 2], 'b-', label='Input History')
    ax.scatter(input_history_pos[-1, 0], input_history_pos[-1, 1], input_history_pos[-1, 2], c='b', marker='o', s=60, label='Last Known Point')
    ax.plot(true_future_pos[:, 0], true_future_pos[:, 1], true_future_pos[:, 2], 'g--', label='True Future')
    ax.plot(predicted_future_pos[:, 0], predicted_future_pos[:, 1], predicted_future_pos[:, 2], 'r-o', markersize=4, label='Predicted Trajectory')

    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_zlabel('Z Coordinate (meters)')
    ax.set_title(f'Trajectory Prediction for Sample #{SAMPLE_INDEX}')
    ax.legend()
    ax.grid(True)
    all_points = np.vstack([input_history_pos, true_future_pos, predicted_future_pos])
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.show()
