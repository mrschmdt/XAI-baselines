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
    # max_distance = 0
    # furthest_baseline = None
    # for i in range(len(dataset)):
    #     distance = torch.norm(x - dataset[i][0])
    #     if distance > max_distance:
    #         max_distance = distance
    #         furthest_baseline = dataset[i][0]

    # print(f"Furthest Baseline: {furthest_baseline}")

    distances = torch.cdist(x.unsqueeze(0), dataset[:][0])  # Compute pairwise distances between the target tensor and each tensor in the set
    max_distance, max_index = torch.max(distances, dim=1)  # Find the tensor with the maximum distance
    furthest_tensor = dataset[max_index][0].squeeze(0)
    print(f"max Baseline: {furthest_tensor}")
    return furthest_tensor

def get_closest_baseline(
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
    min_distance = float('inf')
    closest_baseline = None
    for i in range(len(dataset)):
        distance = torch.norm(x - dataset[i][0])
        if distance < min_distance and distance != 0:
            min_distance = distance
            closest_baseline = dataset[i][0]
    return closest_baseline
