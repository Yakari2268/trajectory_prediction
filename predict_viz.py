import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import your two helper functions
from extract_bag_data import extract_and_sample_bag
from predict_from_coords import predict_from_coordinates

def visualize_trajectory(input_coords, output_coords, predicted_coords):
    """
    Plots input, true output, and predicted trajectory in 3D.
    """
    input_coords = np.array(input_coords)
    output_coords = np.array(output_coords)
    predicted_coords = np.array(predicted_coords)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot input
    ax.plot(input_coords[:, 0], input_coords[:, 1], input_coords[:, 2],
            color='blue', marker='o', label='Input History')

    # Plot true future trajectory
    ax.plot(output_coords[:, 0], output_coords[:, 1], output_coords[:, 2],
            color='green', marker='^', linestyle='--', label='True Future')

    # Plot predicted future trajectory
    ax.plot(predicted_coords[:, 0], predicted_coords[:, 1], predicted_coords[:, 2],
            color='red', marker='s', linestyle='-', label='Predicted Future')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Drone Trajectory Prediction')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    # --- STEP 1: Extract coordinates from .bag file ---
    bag_path = '2023-08-24-10-49-05_mavic2.bag'
    topic_name = '/leica/point/relative'

    input_coords, output_coords = extract_and_sample_bag(
        bag_path=bag_path,
        topic_name=topic_name,
        input_duration_sec=2.0,
        output_duration_sec=1.0,
        sample_rate_hz=10.0,
        start_time_sec=100.0,
        apply_filter=True
    )

    print(f"Extracted {len(input_coords)} input points.")
    print(f"Extracted {len(output_coords)} output points.")

    # --- STEP 2: Predict future trajectory ---
    predicted_coords = predict_from_coordinates(
        drone_coords=input_coords,
        model_path='best_model.pth',
        stats_path='vel_stats.npz',
        output_seq_len=10,  # predict 10 points
        dt=0.1,
        use_velocity_prediction=True,
        use_whitening=False
    )

    print(f"Predicted {len(predicted_coords)} future points.")

    # --- STEP 3: Visualize ---
    visualize_trajectory(input_coords, output_coords, predicted_coords)
