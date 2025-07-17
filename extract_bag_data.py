import bagpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the filter function
from filter_trajectory import filter_trajectory

def extract_and_sample_bag(
    bag_path,
    topic_name,
    input_duration_sec=2.0,
    output_duration_sec=1.0,
    sample_rate_hz=10.0,
    start_time_sec=0.0,
    apply_filter=True
):
    """
    Extracts drone coordinates from a ROS bag file, samples them, and optionally filters noise.

    Parameters
    ----------
    bag_path : str
        Path to the .bag file.
    topic_name : str
        Topic to extract.
    input_duration_sec : float
        Duration of input segment.
    output_duration_sec : float
        Duration of output segment.
    sample_rate_hz : float
        Sampling frequency.
    start_time_sec : float
        Offset into the bag to start extraction.
    apply_filter : bool
        Whether to filter the trajectory.

    Returns
    -------
    input_coords : np.ndarray (N,3)
    output_coords : np.ndarray (M,3)
    """

    # Read bag
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic(topic_name)
    df = pd.read_csv(csv_file)

    # Check required columns
    expected_cols = ['Time', 'point.x', 'point.y', 'point.z']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in bag data.")

    start_time_global = df['Time'].iloc[0]
    df['t_rel'] = df['Time'] - start_time_global
    df = df.sort_values('t_rel')

    dt = 1.0 / sample_rate_hz

    # Input slice
    input_start_time = start_time_sec
    input_end_time = input_start_time + input_duration_sec

    df_input = df[(df['t_rel'] >= input_start_time) & (df['t_rel'] <= input_end_time)]

    t_input = np.arange(input_start_time, input_end_time, dt)
    input_sampled = []
    for t in t_input:
        idx = (df_input['t_rel'] - t).abs().idxmin()
        row = df_input.loc[idx]
        input_sampled.append([row['point.x'], row['point.y'], row['point.z']])
    input_sampled = np.array(input_sampled)

    # Output slice — starts right after input ends
    output_start_time = input_end_time
    output_end_time = output_start_time + output_duration_sec

    df_output = df[(df['t_rel'] >= output_start_time) & (df['t_rel'] <= output_end_time)]

    t_output = np.arange(output_start_time, output_end_time, dt)
    output_sampled = []
    for t in t_output:
        idx = (df_output['t_rel'] - t).abs().idxmin()
        row = df_output.loc[idx]
        output_sampled.append([row['point.x'], row['point.y'], row['point.z']])
    output_sampled = np.array(output_sampled)

    # Apply filtering if enabled
    if apply_filter:
        input_sampled = filter_trajectory(input_sampled)
        output_sampled = filter_trajectory(output_sampled)

    return input_sampled, output_sampled


if __name__ == '__main__':
    bag_file_path = '2023-08-24-10-49-05_mavic2.bag'
    topic = '/leica/point/relative'

    input_pts, output_pts = extract_and_sample_bag(
        bag_path=bag_file_path,
        topic_name=topic,
        input_duration_sec=2.0,
        output_duration_sec=1.0,
        sample_rate_hz=10.0,
        start_time_sec=35.0,
        apply_filter=True
    )

    print("INPUT TRAJECTORY POINTS:")
    print(input_pts)

    print("\nOUTPUT TRAJECTORY POINTS:")
    print(output_pts)

    # --- Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(input_pts[:, 0], input_pts[:, 1], input_pts[:, 2],
            color='blue', marker='o', label='Input History')

    ax.plot(output_pts[:, 0], output_pts[:, 1], output_pts[:, 2],
            color='green', marker='^', linestyle='--', label='Output Future')

    ax.scatter(input_pts[-1, 0], input_pts[-1, 1], input_pts[-1, 2],
               color='black', marker='x', s=80, label='Last Input Point')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Drone Trajectory from ROS Bag')
    ax.legend()
    ax.grid(True)
    plt.show()
