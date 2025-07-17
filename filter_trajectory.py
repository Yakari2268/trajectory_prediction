# filter_trajectory.py

from scipy.signal import savgol_filter
import numpy as np

def filter_trajectory(points, method='savgol', window_size=5, polyorder=2):
    """
    Filters a 3D trajectory using the specified method.
    
    points: ndarray (N,3)
    returns: ndarray (N,3)
    """
    if method == 'savgol':
        points_filtered = savgol_filter(points, window_length=window_size, polyorder=polyorder, axis=0, mode='nearest')
        return points_filtered
    else:
        return points
