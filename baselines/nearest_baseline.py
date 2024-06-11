from baselines import Baseline
from torch.utils.data import Dataset
import torch
from baselines.uniform_output.nearest_furthest_calculator import get_nearest_baseline

class NearestBaseline(Baseline):
    def __init__(self,
        dataset: Dataset
    ) -> None:
        self.dataset = dataset

    def get_baseline(self,
        **kwargs
    ) -> torch.Tensor:
        nearest_baseline = get_nearest_baseline(
            x=kwargs["x"],
            dataset=self.dataset
        )

        return nearest_baseline