from network import NeuralNetwork
from model import CombinedBaselineNetwork, combined_model_loss_fn
import torch
from torch.utils.data import DataLoader
import tqdm



def train_autobaseline(
        classification_model : NeuralNetwork, 
        initial_baseline : torch.Tensor = None, 
        num_epochs : int = 300,
        baseline_error_weight = 0.4
        ) -> torch.Tensor:
    
    """
    This method trains the CombinedBaselineNetwork. To be specific, only trains the first Layer of the CombinedBaselineNetwork consisting of the AutobaselineNetwork. The other
    layers are equivalent to the Dry Beans Network.

    Args:
        classification_model (NeuralNetwork): The traines Dry Beans Model.
        initial_baseline (torch.Tensor): The initial baseline from which the Uniform Output Baseline gets predicted.
        num_epochs (int): Number of epochs to train.
        baseline_error_weight (float): Weight of the baseline error in the loss function.
    """
    
    if initial_baseline == None:
        torch.manual_seed(48)
        n_inputs = classification_model.get_number_of_input_features()
        initial_baseline = torch.FloatTensor(n_inputs).uniform_(0.005,0.01)

    dataset_len = 1000
    x = torch.ones((dataset_len, 1)) #dataset_len x 1
    y_baseline = torch.unsqueeze(initial_baseline,0).repeat(dataset_len,1) #dataset_len x len(initial_baseline)
    y_model_output = torch.ones((1000,7)) * (1/7)

    dataset = torch.utils.data.TensorDataset(x,y_baseline,y_model_output)
    dataloader = DataLoader(dataset=dataset,batch_size=32) #since the dataset consists of the same datapoints, changing the batch_size will only effect the number of training steps.
    combined_baseline_model = CombinedBaselineNetwork(classification_model=classification_model,initial_baseline=initial_baseline)
    optimizer = torch.optim.Adam(params=combined_baseline_model.autobaseline_model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        for x, y_baseline,y_model_output in dataloader:
            optimizer.zero_grad()
            autobaseline, model_output = combined_baseline_model(x)

            loss = combined_model_loss_fn(
                autobaseline=autobaseline,
                initial_baseline=y_baseline,
                actual_model_output=model_output,
                target_model_output=y_model_output,
                baseline_error_weight=baseline_error_weight
            )

            loss.backward()
            optimizer.step()

    print("autobaseline: " + str(combined_baseline_model.get_autobaseline()))
    print("prediction: " + str(classification_model.predict(combined_baseline_model.get_autobaseline())))

    return combined_baseline_model.get_autobaseline().detach()