from .uniform_output import train_autobaseline
import torch
from network import NeuralNetwork
from .baseline import Baseline
from .uniform_output.nearest_furthest_calculator import get_furthest_baseline, get_nearest_baseline, get_furthest_datapoints_of_training_set, get_nearest_datapoints_of_training_set
from tqdm import tqdm

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
        dataset_train: torch.utils.data.Dataset,
        dataset_test: torch.utils.data.Dataset,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:
        
        self.classification_model: NeuralNetwork = classification_model
        self.dataset_train: torch.utils.data.Dataset = dataset_train
        self.dataset_test: torch.utils.data.Dataset = dataset_test
        self.num_epochs: int = num_epochs
        self.baseline_error_weight: float = baseline_error_weight


        self.furthest_datapoint_of_trainings_set: list[torch.Tensor] = get_furthest_datapoints_of_training_set(
            dataset_train=self.dataset_train,
            dataset_test=self.dataset_test
        )

        self.baselines = torch.zeros(len(self.dataset_test), self.dataset_test[0][0].shape[0])

        for i in tqdm(range(len(self.dataset_test))):
            self.baselines[i] = train_autobaseline(
                classification_model=self.classification_model,
                initial_baseline=self.furthest_datapoint_of_trainings_set[i],
                num_epochs=self.num_epochs,
                baseline_error_weight=self.baseline_error_weight
            )
            
    
    def get_baseline(self,
        **kwargs
    )->torch.Tensor:
        
        return self.baselines[kwargs["i"]]
    
class NearestUniformOutputBaseline(Baseline):
    def __init__(self,
        classification_model: NeuralNetwork,
        dataset_train: torch.utils.data.Dataset,
        dataset_test: torch.utils.data.Dataset,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:
        
        self.classification_model: NeuralNetwork = classification_model
        self.dataset_train: torch.utils.data.Dataset = dataset_train
        self.dataset_test: torch.utils.data.Dataset = dataset_test
        self.num_epochs: int = num_epochs
        self.baseline_error_weight: float = baseline_error_weight


        self.nearest_datapoint_of_trainings_set: list[torch.Tensor] = get_nearest_datapoints_of_training_set(
            dataset_train=self.dataset_train,
            dataset_test=self.dataset_test
        )

        self.baselines = torch.zeros(len(self.dataset_test), self.dataset_test[0][0].shape[0])

        for i in tqdm(range(len(self.dataset_test))):
            self.baselines[i] = train_autobaseline(
                classification_model=self.classification_model,
                initial_baseline=self.nearest_datapoint_of_trainings_set[i],
                num_epochs=self.num_epochs,
                baseline_error_weight=self.baseline_error_weight
            )

    def get_baseline(self,
        **kwargs
    )->torch.Tensor:
        
        return self.baselines[kwargs["i"]]

        
        
    
    