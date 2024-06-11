from torch.utils.data import Dataset
import torch

def get_furthest_baseline(
        x: torch.Tensor,
        dataset: Dataset
    )->torch.Tensor:
    """
    Calculate the furthest baseline from the dataset.
    
    Args:
        x: The input tensor.
        dataset: The dataset.

    Returns:
        The furthest baseline.
    """

    distances = torch.cdist(x.unsqueeze(0), dataset[:][0])  # Compute pairwise distances between the target tensor and each tensor in the set
    max_distance, max_index = torch.max(distances, dim=1)  # Find the tensor with the maximum distance
    furthest_tensor = dataset[max_index][0].squeeze(0)
    return furthest_tensor

def get_nearest_baseline(
        x: torch.Tensor,
        dataset: Dataset
    )->torch.Tensor:
    """
    Calculate the closest baseline from the dataset.
    
    Args:
        x: The input tensor.
        dataset: The dataset.

    Returns:
        The closest baseline.
    """
    distances = torch.cdist(x.unsqueeze(0), dataset[:][0])  # Compute pairwise distances between the target tensor and each tensor in the set
    min_distance, min_index = torch.min(distances, dim=1)  # Find the tensor with the maximum distance
    closest_tensor = dataset[min_index][0].squeeze(0)
    return closest_tensor

def get_furthest_datapoints_of_training_set(
        dataset_train: Dataset,
        dataset_test: Dataset
)->list[torch.Tensor]:
    """
    For each test datapoint, find the training sample that is furthest away.
    
    Args:
        dataset_train: The training dataset.
        dataset_test: The test dataset.

    Returns:
        The furthest datapoint.
    """
    distances = torch.cdist(dataset_test[:][0], dataset_train[:][0])
    furthest_distances, furthest_indices = torch.max(distances, dim=1)

    return [dataset_train[i][0] for i in furthest_indices]

def get_nearest_datapoints_of_training_set(
        dataset_train: Dataset,
        dataset_test: Dataset
)->list[torch.Tensor]:
    """
    For each test datapoint, find the training sample that is nearest.
    
    Args:
        dataset_train: The training dataset.
        dataset_test: The test dataset.

    Returns:
        The nearest datapoint.
    """
    distances = torch.cdist(dataset_test[:][0], dataset_train[:][0])
    nearest_distances, nearest_indices = torch.min(distances, dim=1)

    return [dataset_train[i][0] for i in nearest_indices]