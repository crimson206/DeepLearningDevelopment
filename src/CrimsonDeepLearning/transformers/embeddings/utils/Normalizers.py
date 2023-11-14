import torch
import numpy as np

class CategoricalNormalizer2dArray:
    """
    A class for normalizing categorical features in a NumPy array by subtracting the minimum value per feature
    and records the maximum value for setting embedding dimensions.
    
    Attributes:
        min_values (numpy.ndarray): An array holding the minimum values for each categorical feature.
        max_values (numpy.ndarray): An array holding the maximum values for each categorical feature.
    """
    
    def __init__(self):
        """
        Initializes the CategoricalNormalizer instance with no minimum or maximum values set.
        """
        self.min_values = None
        self.max_values = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the normalizer to the data by finding the minimum and maximum values for each feature
        and then normalizes the data by subtracting the minimum values.
        
        Args:
            X (np.ndarray): The input array containing categorical features to normalize.
                            Shape: (n_samples, n_features), where
                            n_samples is the number of samples,
                            n_features is the number of categorical features.

        Returns:
            np.ndarray: The normalized array with the same shape as the input array.
                        Shape: (n_samples, n_features)
        """
        self.min_values = X.min(axis=0)
        self.max_values = X.max(axis=0)  # Store the maximum values
        return X - self.min_values

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalizes the data by subtracting the minimum values found during fitting.
        
        Args:
            X (np.ndarray): The input array containing categorical features to normalize.
                            Must have the same number of features as the data used in `fit_transform`.
                            Shape: (n_samples, n_features)

        Returns:
            np.ndarray: The normalized array with the same shape as the input array.
                        Shape: (n_samples, n_features)

        Raises:
            ValueError: If the normalizer has not been fit yet.
        """
        if self.min_values is None:
            raise ValueError("The normalizer has not been fit yet.")
        return X - self.min_values

    def get_feature_range(self) -> np.ndarray:
        """
        Gets the range of the features based on the max and min values found during fitting.

        Returns:
            np.ndarray: The range of each feature, used for setting embedding dimensions.
                        Shape: (n_features,)
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError("The normalizer has not been fit yet.")
        return self.max_values - self.min_values



class CategoricalNormalizer3dTensor:
    """
    A class for normalizing categorical features in a tensor by subtracting the minimum value.
    
    Attributes:
        min_values (torch.Tensor): A tensor holding the minimum values for each categorical feature.
                                   This tensor is fit during the `fit_transform` method and is used
                                   for normalizing subsequent tensors in the `transform` method.
    """
    
    def __init__(self):
        """
        Initializes the CategoricalNormalizer instance with no minimum values set.
        """
        self.min_values = None

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fits the normalizer to the data by finding the minimum values across all dimensions
        except the last one and then normalizes the data by subtracting these minimum values.
        
        Args:
            X (torch.Tensor): The input tensor containing categorical features to normalize.
                              Shape: (n_batch, n_seq, n_features), where
                              n_batch is the batch size,
                              n_seq is the sequence length,
                              n_features is the number of categorical features.

        Returns:
            torch.Tensor: The normalized tensor with the same shape as the input tensor.
                          Shape: (n_batch, n_seq, n_features)
        """
        # Calculate the minimum values across all batches and sequences for each feature
        self.min_values = X.min(dim=0)[0].min(dim=0)[0]
        # Reshape to allow broadcast subtraction across the batches and sequences
        self.min_values = self.min_values.view(1, 1, -1)
        # Subtract the minimum values from the input tensor to normalize
        return X - self.min_values

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data by subtracting the minimum values found during fitting.
        The normalizer must have been fit to data beforehand using the `fit_transform` method.
        
        Args:
            X (torch.Tensor): The input tensor containing categorical features to normalize.
                              Must have the same last dimension as the data used in `fit_transform`.
                              Shape: (n_batch, n_seq, n_features)

        Returns:
            torch.Tensor: The normalized tensor with the same shape as the input tensor.
                          Shape: (n_batch, n_seq, n_features)

        Raises:
            ValueError: If the normalizer has not been fit yet.
        """
        if self.min_values is None:
            raise ValueError("The normalizer has not been fit yet.")
        # Subtract the minimum values from the input tensor to normalize
        return X - self.min_values
