import torch
import numpy as np

from typing import Callable
from itertools import combinations
from tqdm import tqdm

from network.models import NeuralNetwork
from evaluation.utils.visualisation import _visualize_log_odds, _visualize_log_odds_comparison
from OpenXAI.openxai.dataloader import TabularDataLoader
from data import HELOC


import copy



class AttributionMethodsEvaluator():

    """
    This class serves the evaluation and comparison of the three attribution methods Integrated Gradients, Lime and KernelShap.
    """

    def __init__(self,
        model: NeuralNetwork,
        dataset: str):
        self.model = model
        self.dataset = TabularDataLoader(
            path="Synthetic",
            filename="test",
            label="y",
            scale="minmax")

    def get_log_odds_of_datapoint(
            self,
            x,
            attribute,
            apply_log: bool = True,
            masking_baseline = torch.zeros(20),
            **kwargs) -> np.ndarray:
        """
        Calculates the log odds of one datapoint. Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            x (torch.Tensor): Input for which the Log Odds are being calculated.
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments

                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.

        Returns:
            certainties or log_odds_of_datapoint(np.ndarray): Depending on apply_log
        """
        x = x.copy() #to avoid manipulation the dataset

        target_label_index = self.model.predict(torch.from_numpy(x)).argmax().item()

        inputs = torch.tensor(x, requires_grad=True).unsqueeze(0)

        attribution_scores = attribute(inputs=inputs,target=target_label_index)
        masking_order = torch.argsort(attribution_scores, descending=True)
        masking_order = masking_order.numpy()

        predictions_with_mask = np.zeros((20))

        for i in range(len(predictions_with_mask)):
            x[masking_order[0:i]] = masking_baseline[masking_order[0:i]]
            prediction = self.model.predict(torch.from_numpy(x))
            predictions_with_mask[i] = prediction[target_label_index]

        if apply_log:
            counter_propability = 1 - predictions_with_mask
            log_odds = np.log((predictions_with_mask + 10**(-16)) / counter_propability)
            return log_odds
        else:
            return predictions_with_mask
    
    def get_random_references_of_datapoint(
            self,
            x,
            apply_log : bool = True,
            masking_baseline = torch.zeros(20),
            **kwargs
            ) -> np.ndarray:
        """
        Calculates a random reference of a datapoint by randomly choosing the masking order.

        Args:
            x (torch.Tensor): datapoint for which the reference gets calculated.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default

        Returns:
            certainties or log_odds
        """
        x = x.copy() #to avoid manipulation the dataset
        
        target_label_index = self.model.predict(torch.from_numpy(x)).argmax().item()
        # print("Model prediction: " + str(self.model.predict(x)))
        random_masking_order = np.random.choice(a=20, size=20, replace=False)
        
        predictions_with_random_mask = np.zeros((20))
        # print("initial x: " + str(x))

        for i in range(len(predictions_with_random_mask)):
            x[random_masking_order[0:i]] = masking_baseline[random_masking_order[0:i]]
            # print("Masked input: " + str(x))
            prediction = self.model.predict(torch.from_numpy(x))
            # print("Target-label-index: " + str(target_label_index))
            predictions_with_random_mask[i] = prediction[target_label_index]
            # print("Prediction: " + str(prediction))

        if apply_log:
            counter_propability = 1 - predictions_with_random_mask
            log_odds = np.log(predictions_with_random_mask + 1**(-16) / counter_propability)
            return log_odds
        
        else:
            return predictions_with_random_mask
        
    def get_log_odds_of_masked_weight_datapoint(
            self,
            x: torch.Tensor,
            ground_truth: torch.Tensor,
            apply_log: bool,
            masking_baseline: torch.Tensor,
    )-> torch.Tensor:
        x = x.copy() #to avoid manipulation the dataset

        target_label_index = self.model.predict(torch.from_numpy(x)).argmax().item()
        ground_truth = np.absolute(ground_truth)

        masking_order = torch.argsort(torch.tensor(ground_truth), descending=True)
        masking_order = masking_order.numpy()

        predictions_with_mask = np.zeros((20))

        for i in range(len(predictions_with_mask)):
            x[masking_order[0:i]] = masking_baseline[masking_order[0:i]]
            prediction = self.model.predict(torch.from_numpy(x))
            predictions_with_mask[i] = prediction[target_label_index]

        if apply_log:
            counter_propability = 1 - predictions_with_mask
            log_odds = np.log((predictions_with_mask + 10**(-16)) / counter_propability)
            return log_odds
        else:
            return predictions_with_mask
        
    def get_log_odds_of_binary_mask_datapoint(
            self,
            x: np.ndarray,
            binary_mask: np.ndarray,
            apply_log: bool,
            masking_baseline: torch.Tensor
    )-> torch.Tensor:
        
        x_copy = x.copy()
        
        target_label_index = self.model.predict(torch.from_numpy(x)).argmax().item()

        predictions_with_mask=np.zeros(binary_mask.shape)
        predictions_with_mask[0] = self.model.predict(torch.from_numpy(x))[target_label_index]

        masked_indices = np.where(binary_mask == 1)[0]

        for i in range(1,binary_mask.sum()):
            x = x_copy.copy() 
            avg_prediction = 0
            for subset in combinations(masked_indices, i):
                x[list(subset)] = masking_baseline[list(subset)]
                prediction = self.model.predict(torch.from_numpy(x))
                avg_prediction += prediction[target_label_index]

            avg_prediction = avg_prediction / len(list(combinations(masked_indices, i)))
            avg_prediction = avg_prediction.item()
            predictions_with_mask[i] = avg_prediction

        if apply_log:
            counter_propability = 1 - predictions_with_mask
            log_odds = np.log((predictions_with_mask + 10**(-16)) / counter_propability)
            return log_odds
        else:
            return predictions_with_mask
    
    def get_log_odds_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            attribute,
            apply_log: bool = True,
            masking_baseline = torch.zeros(20),
            **kwargs
        ) -> tuple[np.ndarray, #log-odds
              np.ndarray, #mean of log_odds
              np.ndarray, #max of log_odds
              np.ndarray]:  #min of log_odds

        """
        Calculates the log odds of each datapoint in a dataset and mean, max and min of them. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which the log odds get calculated.
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments
                    
                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """
        log_odds = np.zeros((len(dataset.data),20))

        for i in tqdm(range(len(log_odds))):
        #for i in tqdm(range(100)):
            log_odds[i] = self.get_log_odds_of_datapoint(dataset.data[i],attribute=attribute,apply_log=apply_log, masking_baseline=masking_baseline, **kwargs)

        #mean, max and min calculation
        mean = log_odds.mean(axis=0)

        log_odds_sums = log_odds.sum(axis=1)

        max_index = log_odds_sums.argmax()
        max = log_odds[max_index]

        min_index = log_odds_sums.argmin()
        min = log_odds[min_index]



        return log_odds, mean, max, min
    
    def get_random_references_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            apply_log : bool = True,
            masking_baseline = torch.zeros(20),
            **kwargs
        ) -> tuple[np.ndarray, np.ndarray] :
        """
        Calculates a random reference of each datapoint in a dataset and the mean of them. The masking order is choosen ramdomly for each datapoint. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which the baseline gets calculated.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default

        Returns:
            random references (np.ndarray)
            mean (np.ndarray): mean of references
        """

        random_references = np.zeros((len(dataset), 20))

        for i in tqdm(range(len(random_references))):
        #for i in tqdm(range(100)):
            random_references[i] = self.get_random_references_of_datapoint(dataset.data[i], apply_log=apply_log, masking_baseline=masking_baseline, **kwargs)
            

        mean = random_references.mean(axis=0)

        return random_references, mean
    
    def get_log_odds_of_ground_truth_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            ground_truth: torch.Tensor,
            apply_log: bool,
            masking_baseline: torch.Tensor,
            ground_truth_type:str
    ) -> tuple[np.ndarray, #log-odds
              np.ndarray, #mean of log_odds
              np.ndarray, #max of log_odds
              np.ndarray]:
        
        log_odds = np.zeros((len(dataset),20))

        if ground_truth_type == "masked_weights":
            for i in tqdm(range(len(log_odds))):
                log_odds[i] = self.get_log_odds_of_masked_weight_datapoint(dataset[i], ground_truth[i], apply_log, masking_baseline)

        elif ground_truth_type == "binary_mask":
            for i in tqdm(range(len(log_odds))):
                log_odds[i] = self.get_log_odds_of_binary_mask_datapoint(dataset[i], ground_truth[i], apply_log, masking_baseline)

        #mean, max and min calculation
        mean = log_odds.mean(axis=0)

        log_odds_sums = log_odds.sum(axis=1)

        max_index = log_odds_sums.argmax()
        max = log_odds[max_index]

        min_index = log_odds_sums.argmin()
        min = log_odds[min_index]

        return log_odds, mean, max, min

    def visualize_log_odds_of_dataset(
            self,
            attribute,
            title, 
            apply_log: bool = True,
            masking_baseline = torch.zeros(20),
            **kwargs
        ) -> None:
        """
        Calculates the mean, max and min log odds of each datapoint in a dataset and plots them. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments
            
                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            title (String): title of plot visualizing the log odds
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """

        dataset_copy = copy.deepcopy(self.dataset)
        log_odds, mean, max, min = self.get_log_odds_of_dataset(dataset_copy,attribute,apply_log,masking_baseline,**kwargs)

        dataset_copy = copy.deepcopy(self.dataset)
        random_references, random_references_mean = self.get_random_references_of_dataset(dataset=dataset_copy,apply_log=apply_log,masking_baseline=masking_baseline, **kwargs)

        _visualize_log_odds(title, log_odds, mean, max, min, random_references_mean,apply_log)

    def visualize_log_odds_of_ground_truth_masked_weights(
            self,
            title,
            apply_log: bool,
            masking_baseline = torch.zeros(20),
            **kwargs
    ):
        dataset_copy = copy.deepcopy(self.dataset)
        ground_truth = dataset_copy.masked_weights

        log_odds, mean, max, min = self.get_log_odds_of_ground_truth_dataset(dataset_copy.data, ground_truth, apply_log, masking_baseline,"masked_weights")

        dataset_copy = copy.deepcopy(self.dataset)
        random_references, random_references_mean = self.get_random_references_of_dataset(dataset=dataset_copy,apply_log=apply_log,masking_baseline=masking_baseline, **kwargs)

        _visualize_log_odds(title, log_odds, mean, max, min, random_references_mean,apply_log)


    def visualize_log_odds_of_ground_truth_binary_mask(
            self,
            title,
            apply_log,
            masking_baseline = torch.zeros(20),
            **kwargs
    ):
        dataset_copy = copy.deepcopy(self.dataset)
        ground_truth = dataset_copy.masks

        log_odds, mean, max, min = self.get_log_odds_of_ground_truth_dataset(dataset_copy.data, ground_truth, apply_log, masking_baseline,"binary_mask")


        dataset_copy = copy.deepcopy(self.dataset)
        random_references, random_references_mean = self.get_random_references_of_dataset(dataset=dataset_copy,apply_log=apply_log,masking_baseline=masking_baseline, **kwargs)

        _visualize_log_odds(title, log_odds, mean, max, min, random_references_mean,apply_log)
