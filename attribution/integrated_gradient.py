from captum.attr import IntegratedGradients as IntegratedGradientsCaptum
import torch

class IntegratedGradient:
    def __init__(self, forward_func: callable):
        self.forward_func = forward_func

    def attribute(self,
        input: torch.Tensor,
        baseline: torch.Tensor):
        
        target_index = self.forward_func(input).argmax()
        ig = IntegratedGradientsCaptum(self.forward_func)
        attributions, delta = ig.attribute(
            inputs = input,
            baselines = baseline, 
            target=target_index,
            return_convergence_delta=True)

        return attributions
