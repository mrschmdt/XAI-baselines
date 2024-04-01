from .baseline import Baseline
import torch

class ZeroBaseline(Baseline):
    def get_baseline(self)->torch.Tensor:
        return torch.zeros(self.num_inputs)