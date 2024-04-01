from torch import nn
import torch
from network.models import NeuralNetwork
import copy

class AutoBaselineNetwork(nn.Module):
    """
    Neural Network for autobaseline calculation. This is one part of the Network to calculate the Uniform Output Baseline (see CombinedBaselineNetwork).
    """

    def __init__(self, initial_baseline : torch.Tensor):
        """
        Args:
            initial_baseline (torch.Tensor): Baseline to start from.    
        """
        super().__init__()

        self.model = nn.Linear(1,16,bias=False)
        initial_baseline_with_grad = copy.deepcopy(initial_baseline).requires_grad_(True)
        with torch.no_grad():
            self.model.weight[:,0] = initial_baseline_with_grad
    def forward(self,x):
        return self.model(x)
    

class CombinedBaselineNetwork(nn.Module):
    """
    Neural Network to determine a Uniform Output Baseline. In this network we are only training the Parameters of the autobaseline_model.
    """

    def __init__(self, classification_model : NeuralNetwork, initial_baseline : torch.Tensor):
        super().__init__()

        self.autobaseline_model = AutoBaselineNetwork(initial_baseline=initial_baseline)
        self.classification_model = copy.deepcopy(classification_model)
        self.classification_model.requires_grad_(False)

    def forward(self,x):
        autobaseline_model_output = self.autobaseline_model(x)
        classification_model_output = self.classification_model.predict(autobaseline_model_output, detach=False)

        return autobaseline_model_output,classification_model_output
    
    def get_autobaseline(self):
        return self.autobaseline_model(torch.ones((1)))
    
def combined_model_loss_fn(
        autobaseline : torch.Tensor, 
        initial_baseline : torch.Tensor, 
        actual_model_output : torch.Tensor, 
        target_model_output : torch.Tensor, 
        baseline_error_weight : float):
    
    """
        Loss Function for the CombindedBaselineNetwork.

        Args:
            autobaseline (torch.Tensor): Baseline computed from the AutoBaselineNetwork.
            initial_baseline (torch.Tensor): Initial baseline set before training.
            actual_model_output (torch.Tensor): Model Output of the autobaseline.
            target_model_output (torch.Tensor): Target Model Output of the Baseline. In our case the Uniform distribution.
            baseline_error_weight (float): weight of the baseline error.
    """
    
    l_baseline = torch.nn.functional.l1_loss(autobaseline,initial_baseline)
    l_model_output = torch.nn.functional.l1_loss(actual_model_output, target_model_output)

    return baseline_error_weight * l_baseline + (1-baseline_error_weight) * l_model_output