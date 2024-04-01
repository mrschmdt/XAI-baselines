from .uniform_output import train_autobaseline
import torch
from .baseline import Baseline

class ZeroUniformOutputBaseline(Baseline):

    def get_baseline(self,
            num_epochs=300,
            baseline_error_weight=0.4
    )->torch.Tensor:
        torch.manual_seed(48)
        initial_baseline = torch.FloatTensor(self.num_inputs).uniform_(0.005,0.01)

        return train_autobaseline(
            classification_model=self.classification_model,
            initial_baseline=initial_baseline,
            num_epochs=num_epochs,
            baseline_error_weight=baseline_error_weight
        )
    