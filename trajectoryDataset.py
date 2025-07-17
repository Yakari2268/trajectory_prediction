import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading trajectory segments from an .npz file.
    """
    def __init__(self, npz_file_path):
        """
        Args:
            npz_file_path (str): Path to the .npz file containing the segments.
        """
        data = np.load(npz_file_path)

        # Validate presence of required arrays
        if 'input_segments' not in data or 'output_segments' not in data:
            raise ValueError(
                f"NPZ file missing required arrays. Keys found: {list(data.keys())}"
            )

        # Data is stored as (features, seq_len, num_segments).
        # Transpose to (num_segments, seq_len, features) for PyTorch.
        input_segments = data['input_segments'].transpose(2, 1, 0)
        output_segments = data['output_segments'].transpose(2, 1, 0)

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(input_segments).float()
        self.y = torch.from_numpy(output_segments).float()

    def __len__(self):
        """Returns the total number of segments in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single input/output pair."""
        return self.X[idx], self.y[idx]