import os
import pickle
from .uniform_output_baseline import ZeroUniformOutputBaseline, FurthestUniformOutputBaseline
import torch
from network import NeuralNetwork

BASE_DIR = os.getenv('BASE_DIR')

def get_precomputed_zero_uniform_ouput_baseline()->ZeroUniformOutputBaseline:

    try:
        with open(os.path.join(BASE_DIR, "baselines","precomputed", "zero_uniform_output_baseline.pkl"), "rb") as f:
            return pickle.load(f)
        
    except FileNotFoundError:
        raise FileNotFoundError("No precomputed Zero Uniform Output Baseline found.")
    

def set_precomputed_zero_uniform_output_baseline(
        classification_model: NeuralNetwork,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:

    zero_uniform_output_baseline = ZeroUniformOutputBaseline(classification_model,num_epochs,baseline_error_weight)

    with open(os.path.join(BASE_DIR, "baselines","precomputed", "zero_uniform_output_baseline.pkl"), "wb") as f:
        pickle.dump(zero_uniform_output_baseline, f)


def get_precomputed_furthest_uniform_output_baseline()->FurthestUniformOutputBaseline:
    
        try:
            with open(os.path.join(BASE_DIR, "baselines","precomputed", "furthest_uniform_output_baseline.pkl"), "rb") as f:
                return pickle.load(f)
            
        except FileNotFoundError:
            raise FileNotFoundError("No precomputed Furthest Uniform Output Baseline found.")
        
def set_precomputed_furthest_uniform_output_baseline(
        classification_model: NeuralNetwork,
        dataset_train: torch.utils.data.Dataset,
        dataset_test: torch.utils.data.Dataset,
        num_epochs:int = 300,
        baseline_error_weight:float = 0.4
    )->None:

    furthest_uniform_output_baseline = FurthestUniformOutputBaseline(classification_model,dataset_train,dataset_test,num_epochs,baseline_error_weight)

    with open(os.path.join(BASE_DIR, "baselines","precomputed", "furthest_uniform_output_baseline.pkl"), "wb") as f:
        pickle.dump(furthest_uniform_output_baseline, f
)


    
