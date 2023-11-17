import copy
import torch
from typing import Dict, Tuple, Union

def sliding_ensemble(
    model: torch.nn.Module, 
    input_dict: Dict[str, torch.Tensor], 
    window_size: int, 
    ensemble_depth: int, 
    output_shape: Union[Tuple[int, int], Tuple[int, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform an ensemble prediction over a sequence by sliding a window and averaging the predictions.

    The function takes an input dictionary mapping input names to tensors, slides a window of
    a specified size across the input tensors, and aggregates the predictions from the model
    at each window position.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        input_dict (Dict[str, torch.Tensor]): A dictionary containing input data as tensors.
                                               The tensors are expected to have a sequence dimension.
        window_size (int): The size of the window to slide over the sequence dimension of the input.
        ensemble_depth (int): The number of windows to use for the ensemble.
        output_shape (Union[Tuple[int, int], Tuple[int, int, int]]): The shape of the output tensor.
                                                                     The shape can have 2 or 3 dimensions,
                                                                     but the number of dimensions must be greater than 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                                           - The ensembled predictions as a tensor.
                                           - A tensor representing the count of predictions added at each sequence position.
    """
    if len(output_shape) <= 1:
        raise ValueError("output_shape must have more than one dimension.")

    seq_len = output_shape[1]
    gap = seq_len - window_size

    sum_pred = torch.zeros(output_shape)
    counts = torch.zeros(output_shape)

    start_indexes = [int(i * gap // (ensemble_depth - 1)) for i in range(ensemble_depth)]

    model.eval()

    with torch.no_grad():
        for start_index in start_indexes:
            input_temp = copy.deepcopy(input_dict)
            end_index = start_index + window_size
            
            for key, value in input_temp.items():
                if len(value.shape) == 3:
                    input_temp[key] = value[:, start_index:end_index, :]
                elif len(value.shape) == 2:
                    input_temp[key] = value[:, start_index:end_index]
                else:
                    raise ValueError("Input tensors must be 2D or 3D.")

            output = model(**input_temp)

            if len(output_shape) == 3:
                sum_pred[:, start_index:end_index, :] += output
                counts[:, start_index:end_index, :] += 1
            else:
                sum_pred[:, start_index:end_index] += output
                counts[:, start_index:end_index] += 1

    ensembled_pred = sum_pred / counts

    return ensembled_pred, counts