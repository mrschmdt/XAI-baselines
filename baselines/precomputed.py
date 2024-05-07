import os
import pickle
from .uniform_output_baseline import ZeroUniformOutputBaseline
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
    
