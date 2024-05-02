from .models import NeuralNetwork
import torch
from torch.utils.data import DataLoader
from OpenXAI.openxai.dataloader import return_loaders
from .util.visualization import visualise_loss_and_accuracy
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
from data.datasets import HELOC

def train_model(
        dataset: str,
        layers: list[int],
        gauss_params: dict = None,
        num_epochs: int = 10,
        lr: float = 0.001,
        batch_size= 8
         )->tuple[
             torch.nn.Module,
             list[float],
             list[float],
             list[float],
             list[float]
         ]:
    """
    This method instantiates a NeuralNetwork with the given layers and trains it on the Dry Beans Dataset. 
    It returns the model and four performance metrics. The metrics are calculated before each epoch and after the final epoch.

    Args:
        dataset (str): The name of the dataset.
        layers (list[int]): list of the neuron-numbers of the layers.
        num_epochs (int): Number of epochs the network should be trained.
        lr (int): The learning rate.

    returns:
        model (torch.nn.Module): The trained network on the Dry Beans Dataset.
        test_loss_array (list[float]): The average test-loss per epoch.
        test_accuracy_array (list[float]): The average test-accuracy per epoch.
        train_loss_array (list[float]): The average train-loss per epoch.
        train_accuracy_array (list[float]): The average train-accuracy per epoch.
    
    """

    if dataset == 'synthetic':

        train_dataloader, test_dataloader = return_loaders(
            data_name='synthetic',
            batch_size=batch_size,
            gauss_params=gauss_params
        )
    
    elif dataset == 'heloc':
        train_dataset = HELOC(mode='train')
        test_dataset = HELOC(mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=10,average="macro")
    test_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=10,average="macro")
    test_accuracy_per_class = torchmetrics.Accuracy(task="multiclass",num_classes=10,average=None)




    torch.manual_seed(42)
    model = NeuralNetwork(layers=layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = [], [], [], []

    def _train_and_test_one_epoch(train: bool):
        #test-loss and accuracy
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for test_inputs,_,_,_,_,_,test_labels in test_dataloader:
                test_predictions = model(test_inputs)
                test_loss += loss_fn(test_predictions, test_labels).item()
                test_accuracy.update(test_predictions, test_labels)
                test_accuracy_per_class.update(test_predictions,  test_labels)

        test_loss = test_loss / len(test_dataloader)

        #training-loss and accuracy and training
        train_loss = 0
        for train_inputs,_,_,_,_,_, train_labels in train_dataloader:
            #train error
            model.eval()
            with torch.no_grad():
                train_predictions = model(train_inputs)
                train_loss += loss_fn(train_predictions, train_labels).item()
                train_accuracy.update(train_predictions, train_labels)
        for train_inputs,_,_,_,_,_,train_labels in train_dataloader:
            if train:
                #training
                model.train()
                optimizer.zero_grad()
                train_predictions = model(train_inputs)
                loss = loss_fn(train_predictions, train_labels)
                loss.backward()
                optimizer.step()
            
        train_loss = train_loss / len(train_dataloader)

        test_loss_array.append(test_loss)
        test_accuracy_array.append(test_accuracy.compute().item())

        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy.compute().item())
        plt.show()

    for epoch in tqdm(range(num_epochs)):
        _train_and_test_one_epoch(train=True)
        

    #to get the train and test errors as well as accuracy after the last train epoch
    _train_and_test_one_epoch(train=False)
    print("Final metrics: ")
    print(f"Validation-Loss: {test_loss_array[-1]: 0.3f}")
    print(f"Validation-Accuracy: {test_accuracy_array[-1]: 0.1%}")
    print(f"train-Loss: {train_loss_array[-1]: 0.3f}")
    print(f"train-Accuracy: {train_accuracy_array[-1]: 0.1%}")
    total_params = sum(param.numel() for param in model.parameters())
    print("# Parameters: " + str(total_params))
    print(model)

    return model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array

def train_model_and_visualize(
        dataset: str,
        layers: list[int],
        gauss_params: dict = None,
        num_epochs: int = 10,
        lr: float = 0.001,
        batch_size= 8,
        ) -> torch.nn.Module:
    model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = train_model(
        dataset=dataset,
        layers=layers,
        gauss_params=gauss_params,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size)

    visualise_loss_and_accuracy(
        train_accuracy=train_accuracy_array,
        train_loss=train_loss_array,
        validation_accuracy=test_accuracy_array,
        validation_loss=test_loss_array
    )
    return model
