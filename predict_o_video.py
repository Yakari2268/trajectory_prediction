import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- 1. Model ---
class TrajectoryPredictor(nn.Module):
    """Encoder-decoder GRU model for UAV trajectory prediction."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.decoder_gru = nn.GRU(
            input_size=output_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
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
            decoder_input = prediction.unsqueeze(1)  # autoregressive
        return torch.cat(outputs, dim=1)

# --- 2. Helpers ---
def denormalize_data(data, stats, use_whitening):
    if use_whitening:
        mean = stats['mean']
        l_matrix = stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data @ l_inv.T) + mean
    else:  # Max L2-Norm
        if 'max_magnitude' in stats:
            max_norm = stats['max_magnitude']
        elif 'max_length' in stats:
            max_norm = stats['max_length']
        else:
            raise KeyError("Stats missing 'max_magnitude' or 'max_length'")
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    start_pos = np.asarray(start_pos)
    velocities = np.asarray(velocities)
    displacements = np.cumsum(velocities * dt, axis=0)
    return start_pos + displacements

# --- 3. Main ---
if __name__ == '__main__':
    # --- Config ---
    MODEL_PATH = 'best_model_6.pth'
    STATS_PATH = 'vel_stats_4.npz'
    VAL_SEGMENTS_PATH = 'val_segments_6.npz'
    SAMPLE_INDEX = 999

    USE_VELOCITY_PREDICTION = True
    USE_WHITENING = False
    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 3, 128, 3
    NUM_LAYERS, DROPOUT_PROB, DT = 3, 0.5, 0.1

    # --- Load model & data ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"Loaded model: {MODEL_PATH}")

    stats = np.load(STATS_PATH)
    val_data = np.load(VAL_SEGMENTS_PATH)

    # --- Extract one sample ---
    input_sample_raw = val_data['input_segments'][:, :, SAMPLE_INDEX]
    true_output_sample_raw = val_data['output_segments'][:, :, SAMPLE_INDEX]
    input_segment_normalized = input_sample_raw.T
    true_output_segment_normalized = true_output_sample_raw.T

    # --- Predict ---
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_segment_normalized).float().unsqueeze(0).to(device)
        future_len = true_output_segment_normalized.shape[0]
        predicted_output_norm = model(input_tensor, future_len=future_len)
        predicted_output_norm_np = predicted_output_norm.cpu().squeeze(0).numpy()

        input_history_denorm = denormalize_data(input_segment_normalized, stats, USE_WHITENING)
        true_future_denorm = denormalize_data(true_output_segment_normalized, stats, USE_WHITENING)
        predicted_future_denorm = denormalize_data(predicted_output_norm_np, stats, USE_WHITENING)

        if USE_VELOCITY_PREDICTION:
            input_history_pos = integrate_velocity(np.zeros(3), input_history_denorm, DT)
            last_pos = input_history_pos[-1]
            true_future_pos = integrate_velocity(last_pos, true_future_denorm, DT)
            predicted_pos = integrate_velocity(last_pos, predicted_future_denorm, DT)
        else:
            input_history_pos = input_history_denorm
            true_future_pos = true_future_denorm
            predicted_pos = predicted_future_denorm

    # --- Animation ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(input_history_pos[:, 0], input_history_pos[:, 1], input_history_pos[:, 2],
            'b-', label='Input History')
    ax.scatter(input_history_pos[-1, 0], input_history_pos[-1, 1], input_history_pos[-1, 2],
               c='b', marker='o', s=60, label='Last Known Point')

    true_line, = ax.plot([], [], [], 'g--', label='True Future')
    pred_line, = ax.plot([], [], [], 'r-o', markersize=4, label='Predicted Trajectory')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Trajectory Prediction (Follow-Cam)')
    ax.legend()
    ax.grid(True)

    def update(frame):
        # Draw partial trajectories
        true_line.set_data(true_future_pos[:frame, 0], true_future_pos[:frame, 1])
        true_line.set_3d_properties(true_future_pos[:frame, 2])

        pred_line.set_data(predicted_pos[:frame, 0], predicted_pos[:frame, 1])
        pred_line.set_3d_properties(predicted_pos[:frame, 2])

        # --- Move camera with drone ---
        if frame < len(predicted_pos):
            drone_pos = predicted_pos[frame]
            elev = 25 + 5*np.sin(frame * 0.1)   # smooth oscillation
            azim = (frame * 4) % 360            # rotating around
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(drone_pos[0]-20, drone_pos[0]+20)
            ax.set_ylim(drone_pos[1]-20, drone_pos[1]+20)
            ax.set_zlim(drone_pos[2]-10, drone_pos[2]+10)

        return true_line, pred_line

    frames = len(true_future_pos)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    ani.save("trajectory_evolution_followcam.mp4", writer="ffmpeg", fps=5)

    plt.close(fig)
    print("Saved trajectory follow-cam video as trajectory_evolution_followcam.mp4")
