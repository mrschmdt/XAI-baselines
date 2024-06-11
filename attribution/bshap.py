from captum.attr import KernelShap
import torch

class BShap:
    def __init__(self, forward_func: callable):
        self.forward_func = forward_func

    def attribute(self,
        input: torch.Tensor,
        baseline: torch.Tensor):
        
        target_index = self.forward_func(input).argmax()
        kernel_shap = KernelShap(self.forward_func)
        attributions = kernel_shap.attribute(
            inputs = input,
            target=target_index,
            baselines=baseline
        )

        return attributions
