import torch

def generate_sliding_window_sequences(data: torch.Tensor, sequence_length: int, step: int, drop_last: bool = False) -> torch.Tensor:
    """
    Generate sequences from 2D data using a sliding window approach.

    Args:
        data (torch.Tensor): The 2D tensor from which to generate sequences. Shape: (n_data, n_features).
        sequence_length (int): The length of each sequence.
        step (int): The number of time steps to slide the window along the data axis.
        drop_last (bool): Whether to drop the last window if it's not equal to the specified sequence length.

    Returns:
        torch.Tensor: A 3D tensor of shape (n_batch, sequence_length, n_features).
    """
    sequences = []
    for start_idx in range(0, data.size(0), step):
        end_idx = start_idx + sequence_length
        if end_idx > data.size(0):
            if not drop_last:
                sequences.append(data[-sequence_length:])  # Take the last sequence_length elements
            break
        sequences.append(data[start_idx:end_idx])
    return torch.stack(sequences)

def split_tensor(tensor):
    split_tensor = [tensor[:,:,i] for i in range(tensor.shape[-1])]
    return split_tensor