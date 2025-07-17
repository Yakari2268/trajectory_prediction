import torch
import torch.nn as nn
import numpy as np

# --- 1. Define the Model ---
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

# --- 2. Utilities ---
def denormalize_data(data, stats, use_whitening):
    if use_whitening:
        mean = stats['mean']
        l_matrix = stats['L_matrix']
        l_inv = np.linalg.inv(l_matrix)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data @ l_inv.T) + mean
    else:
        max_norm = stats.get('max_magnitude', stats.get('max_length'))
        return data * max_norm

def integrate_velocity(start_pos, velocities, dt=0.1):
    start_pos = np.asarray(start_pos)
    velocities = np.asarray(velocities)
    displacements = np.cumsum(velocities * dt, axis=0)
    return start_pos + displacements

# --- 3. Predict Trajectory from List of Coordinates ---
def predict_from_coordinates(
    drone_coords,
    model_path,
    stats_path,
    output_seq_len=10,
    dt=0.1,
    use_velocity_prediction=True,
    use_whitening=False,
    hidden_dim=128,
    num_layers=3,
    dropout_prob=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = output_dim = 3

    # Load model
    model = TrajectoryPredictor(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load statistics
    stats = np.load(stats_path)

    # Convert input to np array
    drone_coords = np.asarray(drone_coords)
    seq_len = drone_coords.shape[0]

    # Convert positions to velocities if needed
    if use_velocity_prediction:
        velocities = np.diff(drone_coords, axis=0) / dt
        velocities = np.vstack([velocities, np.zeros((1, 3))])
        input_features = velocities
    else:
        input_features = drone_coords

    # Normalize
    if use_whitening:
        mean = stats['mean']
        L = stats['L_matrix']
        input_norm = (input_features - mean) @ L.T
    else:
        max_norm = stats.get('max_magnitude', stats.get('max_length'))
        input_norm = input_features / max_norm

    # Model inference
    input_tensor = torch.from_numpy(input_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_output_norm = model(input_tensor, future_len=output_seq_len)

    predicted_output_norm_np = predicted_output_norm.cpu().squeeze(0).numpy()

    # Denormalize output
    predicted_output = denormalize_data(predicted_output_norm_np, stats, use_whitening)

    # Integrate velocities to positions if required
    if use_velocity_prediction:
        last_known_pos = drone_coords[-1]
        predicted_positions = integrate_velocity(last_known_pos, predicted_output, dt)
    else:
        predicted_positions = predicted_output

    # Convert to Python list
    return predicted_positions.tolist()

# --- 4. Example Usage ---
if __name__ == '__main__':
    # Example: replace with your real drone trajectory data
    dummy_coords = np.cumsum(np.random.randn(20, 3) * 0.01, axis=0).tolist()

    predictions = predict_from_coordinates(
        drone_coords=dummy_coords,
        model_path='best_model.pth',
        stats_path='vel_stats.npz',
        output_seq_len=10,
        dt=0.1,
        use_velocity_prediction=True,
        use_whitening=False
    )

    print("Predicted trajectory points:")
    for pt in predictions:
        print(pt)
