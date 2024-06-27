import torch
import numpy as np


def top_k_feature_indices(
    attribution_scores: torch.Tensor,
    k: int,
):
    """
    Returns the indices of the top k features by absolute attribution value.
    """

    attribution_scores_abs = attribution_scores.abs()
    top_k_indices = torch.topk(attribution_scores_abs, k=k).indices

    return top_k_indices

def feature_agreement(
    attribution_scores_1: torch.Tensor,
    attribution_scores_2: torch.Tensor,
    k: int,
):
    """
    Returns the percentage of agreement between the top k features of two attribution score tensors.
    """

    top_k_indices_1 = top_k_feature_indices(attribution_scores_1, k)
    top_k_indices_2 = top_k_feature_indices(attribution_scores_2, k)

    agreement = np.intersect1d(top_k_indices_1, top_k_indices_2).size / k

    return agreement

def feature_rank_agreement(
    attribution_scores_1: torch.Tensor,
    attribution_scores_2: torch.Tensor,
    k: int,
)-> float:
    """
    Computes the fraction of features that are not only common between the sets of top-k features of two explanations, but also have the same position in the respective rank orders.
    """

    top_k_indices_1 = top_k_feature_indices(attribution_scores_1, k)
    top_k_indices_2 = top_k_feature_indices(attribution_scores_2, k)

    torch.zeros_like(top_k_indices_1)
    feature_rank_agreement_bool = top_k_indices_1 == top_k_indices_2
    
    feature_rank_agreement = feature_rank_agreement_bool.sum().item() / k

    return feature_rank_agreement
    