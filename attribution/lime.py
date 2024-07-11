from captum.attr import Lime as LimeCaptum
import torch
import numpy as np

class Lime:
    def __init__(self, forward_func: callable):
        self.forward_func = forward_func

    def attribute(self,
        input: torch.Tensor,
        baseline: torch.Tensor):
        
        target_index = self.forward_func(input).argmax()
        lime = LimeCaptum(self.forward_func)
        attributions = lime.attribute(
            inputs = input,
            baselines = baseline, 
            target=target_index,
            n_samples=100
        )

        return attributions
