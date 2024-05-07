from .uniform_output import train_autobaseline
import torch
from network import NeuralNetwork
from .baseline import Baseline
from .uniform_output.nearest_furthest_calculator import get_furthest_baseline, get_nearest_baseline

class ZeroUniformOutputBaseline(Baseline):

    def __init__(self,
        classification_model: NeuralNetwork,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:
        self.num_inputs = classification_model.get_number_of_input_features()

        torch.manual_seed(48)
        initial_baseline = torch.FloatTensor(self.num_inputs).uniform_(0.005,0.01)

        self.baseline = train_autobaseline(
            classification_model=classification_model,
            initial_baseline=initial_baseline,
            num_epochs=num_epochs,
            baseline_error_weight=baseline_error_weight
        )
        
    def get_baseline(self,
        **kwargs)->torch.Tensor:

        if kwargs:
            pass

        return self.baseline
    
class FurthestUniformOutputBaseline(Baseline):
    def __init__(self,
        classification_model: NeuralNetwork,
        dataset: torch.utils.data.Dataset,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:
        
        self.classification_model: NeuralNetwork = classification_model
        self.dataset: torch.utils.data.Dataset = dataset
        self.num_epochs: int = num_epochs
        self.baseline_error_weight: float = baseline_error_weight

    


    def get_baseline(self,
        **kwargs
    )->torch.Tensor:
        
        furthest_baseline = get_furthest_baseline(
            x=kwargs["x"],
            dataset=self.dataset
        )

        return train_autobaseline(
            classification_model=self.classification_model,
            initial_baseline=furthest_baseline,
            num_epochs=self.num_epochs,
            baseline_error_weight=self.baseline_error_weight
        )
        
        
    
    