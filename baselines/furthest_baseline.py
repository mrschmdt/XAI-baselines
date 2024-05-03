from baselines import Baseline
from torch.utils.data import Dataset
import torch
from baselines.uniform_output.nearest_furthest_calculator import get_furthest_baseline

class FurthestBaseline(Baseline):
    def __init__(self,
        dataset: Dataset
    ) -> None:
        self.dataset = dataset

    def get_baseline(self,
        **kwargs
    ) -> torch.Tensor:
        furthest_baseline = get_furthest_baseline(
            x=kwargs["x"],
            dataset=self.dataset
        )

        return furthest_baseline