from .baseline import Baseline
import torch
from data import HELOC
from torch.utils.data import Dataset

class MeanBaseline(Baseline):

    def __init__(self,
        dataset: Dataset)->None:
        self.dataset = dataset

    def get_baseline(
        self, **kwargs)->torch.Tensor:
        return torch.mean(self.dataset[:][0], dim=0)
        

        