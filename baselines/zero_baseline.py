from .baseline import Baseline
from network import NeuralNetwork
import torch

class ZeroBaseline(Baseline):
    def __init__(self, NeuralNetwork: NeuralNetwork):
        self.num_inputs = NeuralNetwork.get_number_of_input_features()

    def get_baseline(self, **kwargs)->torch.Tensor:
        return torch.zeros(self.num_inputs)