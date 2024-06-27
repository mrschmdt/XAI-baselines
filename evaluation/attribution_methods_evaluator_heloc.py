import torch
import numpy as np

from typing import Callable
from itertools import combinations
from tqdm import tqdm
from .utils import feature_agreement, feature_rank_agreement
from network.models import NeuralNetwork
from evaluation.utils.visualisation import _visualize_log_odds, _visualize_log_odds_comparison, visualize_log_odds_of_attribution_methods, visualize_logs_odds_with_different_masking_baselines
from OpenXAI.openxai.dataloader import TabularDataLoader
from data import HELOC

from baselines import Baseline, ZeroBaseline, ZeroUniformOutputBaseline, FurthestBaseline,NearestBaseline, MeanBaseline
from baselines.precomputed import get_precomputed_furthest_uniform_output_baseline, get_precomputed_nearest_uniform_output_baseline

import copy
import statistics
from attribution import IntegratedGradient, Lime, BShap



class AttributionMethodsEvaluator():

    """
    This class serves the evaluation and comparison of the three attribution methods Integrated Gradients, Lime and KernelShap.
    """

    def __init__(self,
        model: NeuralNetwork,
        dataset: str):
        self.model = model
        
        if dataset == "HELOC":
            self.dataset = HELOC(mode="validation")

        self.baselines_mapping = {
            "zero": ZeroBaseline(self.model),
            "zero_uniform_output": ZeroUniformOutputBaseline(self.model),
            "mean": MeanBaseline(self.dataset),
            "furthest": FurthestBaseline(self.dataset),
            "nearest": NearestBaseline(self.dataset),
            "nearest_uniform_output": get_precomputed_nearest_uniform_output_baseline(),
            "furthest_uniform_output": get_precomputed_furthest_uniform_output_baseline()
        }

        self.attribution_methods = {
            "integrated_gradients": IntegratedGradient(self.model),
            "lime": Lime(self.model),
            "bshap": BShap(self.model)
        }




    def get_log_odds_of_datapoint(
            self,
            x,
            i:int,
            attribute:Callable,
            apply_log:bool,
            attribution_baseline_: Baseline,
            masking_baseline_: Baseline,
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


        attribution_baseline = attribution_baseline_.get_baseline(x=x,i=i)

        if attribution_baseline_ != masking_baseline_: 
            masking_baseline = masking_baseline_.get_baseline(x=x, i=i)
        else:
            masking_baseline = attribution_baseline

        x = torch.clone(x) #to avoid manipulation the dataset

        target_label_index = self.model.predict(x).argmax().item()

        input = torch.tensor(x, requires_grad=True).unsqueeze(0)


        attribution_scores = attribute(input=input,baseline=attribution_baseline.unsqueeze(dim=0)).squeeze(0)
        masking_order = torch.argsort(attribution_scores, descending=True)
    

        predictions_with_mask = torch.zeros_like(x)

        for i in range(len(predictions_with_mask)):
            x[masking_order[0:i]] = masking_baseline[masking_order[0:i]]
            prediction = self.model.predict(x)
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
            i: int,
            masking_baseline_: Baseline,
            apply_log : bool = True,
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
        masking_baseline = masking_baseline_.get_baseline(x=x,i=i)
        x = torch.clone(x) #to avoid manipulation the dataset
        
        target_label_index = self.model.predict(x).argmax().item()

        x_len = len(x)
        random_masking_order = np.random.choice(a=x_len, size=x_len, replace=False)
        
        predictions_with_random_mask = np.zeros((x_len))
        # print("initial x: " + str(x))

        for i in range(len(predictions_with_random_mask)):
            x[random_masking_order[0:i]] = masking_baseline[random_masking_order[0:i]]
            # print("Masked input: " + str(x))
            prediction = self.model.predict(x)
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
            apply_log: bool,
            attribution_baseline: Baseline,
            masking_baseline: Baseline,
            num_samples: int = 100,
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
        if num_samples is None:
            num_samples = len(dataset)


        log_odds = np.zeros((num_samples,len(dataset[0][0])))

        for i in tqdm(range(len(log_odds))):
            log_odds[i] = self.get_log_odds_of_datapoint(dataset[i][0],i=i,attribute=attribute,apply_log=apply_log, attribution_baseline_=attribution_baseline, masking_baseline_=masking_baseline, **kwargs)

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
            masking_baseline = Baseline,
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

        random_references = np.zeros((len(dataset), len(dataset[0][0])))

        for i in tqdm(range(len(random_references))):
        #for i in tqdm(range(100)):
            random_references[i] = self.get_random_references_of_datapoint(dataset[i][0],i, apply_log=apply_log, masking_baseline_=masking_baseline, **kwargs)
            

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
    
    def get_feature_agreement_matrix(
        self,
        k: int,
        rank_agreement: bool = False
    )-> tuple[np.ndarray, np.ndarray]:
        
        distance_matrix_mean = np.zeros((len(self.baselines_mapping.keys())*3, len(self.baselines_mapping.keys())*3))
        distance_matrix_std = np.zeros((len(self.baselines_mapping.keys())*3, len(self.baselines_mapping.keys())*3))
        baselines: list[Baseline] = list(self.baselines_mapping.values())
        print(f"baselines: {baselines}")
        attribution_methods: list = list(self.attribution_methods.values())
        print(f"attribution_methods: {attribution_methods}")

        for baseline_a in tqdm(baselines):
            for index_am_a, am_a in enumerate(attribution_methods):
                for index_baseline_a, baseline_a in enumerate(baselines):
                    for index_am_b,am_b in enumerate(attribution_methods):
                        for index_baseline_b, baseline_b in enumerate(baselines):
                            agreements: list[float] = []
                            for i in range(len(self.dataset)):
                                x = self.dataset[i][0]
                                x = torch.clone(x)
                                input = torch.tensor(x, requires_grad=True).unsqueeze(0)
                                attribution_scores_a = am_a.attribute(input = input, baseline = baseline_a.get_baseline(x=x,i=i).unsqueeze(dim=0)).squeeze(0)
                                attribution_scores_b = am_b.attribute(input = input, baseline = baseline_b.get_baseline(x=x,i=i).unsqueeze(dim=0)).squeeze(0)
                                agreements.append(feature_agreement(attribution_scores_a, attribution_scores_b, k))
                            print(f"last computed: {am_a} {baseline_a} {am_b} {baseline_b} {agreements[-1]}")
                            distance_matrix_mean[index_am_a*3+index_baseline_a][index_am_b*3+index_baseline_b] = statistics.mean(agreements)
                            distance_matrix_std[index_am_a*3+index_baseline_a][index_am_b*3+index_baseline_b] = statistics.stdev(agreements)
                        break
                    break
                break
            break
        return (distance_matrix_mean, distance_matrix_std)



    def visualize_log_odds_of_dataset(
            self,
            attribute,
            title, 
            attribution_baseline : Baseline,
            masking_baseline : Baseline,
            apply_log: bool = True,
            num_samples: int = 100,
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
        log_odds, mean, max, min = self.get_log_odds_of_dataset(dataset_copy,attribute,apply_log,attribution_baseline,masking_baseline,num_samples,**kwargs)
        print("Mean: " + str(mean))

        dataset_copy = copy.deepcopy(self.dataset)
        random_references, random_references_mean = self.get_random_references_of_dataset(dataset=dataset_copy,apply_log=apply_log,attribution_baseline=attribution_baseline,masking_baseline=masking_baseline, **kwargs)

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

    def visualize_log_odds_of_attribution_methods(
            self,
            attribute: Callable,
            apply_log: bool = False,
            num_samples: int = 100,
            title=""
    ):

        test_dataset = HELOC(mode="test")
        zero_baseline = ZeroBaseline(self.model)
        zero_uniform_output_baseline = ZeroUniformOutputBaseline(self.model)
        mean_baseline = MeanBaseline(test_dataset)
        furthest_baseline = FurthestBaseline(test_dataset)
        nearest_baseline = NearestBaseline(test_dataset)
        nearest_uniform_output_baseline = get_precomputed_nearest_uniform_output_baseline()
        furthest_uniform_output_baseline = get_precomputed_furthest_uniform_output_baseline()

        dataset_copy = copy.deepcopy(self.dataset)
        zero_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=zero_baseline,
            masking_baseline=zero_baseline,
            num_samples=num_samples
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        zero_uniform_output_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=zero_uniform_output_baseline,
            masking_baseline=zero_uniform_output_baseline,
            num_samples=num_samples
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        mean_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=mean_baseline,
            masking_baseline=mean_baseline,
            num_samples=num_samples
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        furthest_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=furthest_baseline,
            masking_baseline=furthest_baseline,
            num_samples=num_samples
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        nearest_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=nearest_baseline,
            masking_baseline=nearest_baseline,
            num_samples=num_samples
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        nearest_uniform_output_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=apply_log,
            attribution_baseline=nearest_uniform_output_baseline,
            masking_baseline=nearest_uniform_output_baseline
        )[1]

        dataset_copy = copy.deepcopy(self.dataset)
        furthest_uniform_output_baseline_log_odds_mean = self.get_log_odds_of_dataset(
            dataset=dataset_copy,
            attribute=attribute,
            apply_log=False,
            attribution_baseline=furthest_uniform_output_baseline,
            masking_baseline=furthest_uniform_output_baseline
        )[1]

        visualize_log_odds_of_attribution_methods(
            zero_baseline_log_odds_mean=zero_baseline_log_odds_mean,
            zero_uniform_output_baseline_log_odds_mean=zero_uniform_output_baseline_log_odds_mean,
            mean_baseline_log_odds_mean=mean_baseline_log_odds_mean,
            furthest_baseline_log_odds_mean=furthest_baseline_log_odds_mean,
            nearest_baseline_log_odds_mean=nearest_baseline_log_odds_mean,
            nearest_uniform_output_baseline_log_odds_mean=nearest_uniform_output_baseline_log_odds_mean,
            furthest_uniform_output_baseline_log_odds_mean=furthest_uniform_output_baseline_log_odds_mean,
            apply_log=apply_log,
            title=title
        )

    def get_log_odds_of_attribution_with_all_masking_baselines(
            self,
            attribute: Callable,
            attribution_baseline_str: str,
            apply_log: bool = False,
            num_samples: int = 100
    ):

        attribution_baseline = self.baselines_mapping[attribution_baseline_str]

        log_odds = {}

        for key in self.baselines_mapping.keys():
            masking_baseline = self.baselines_mapping[key]
            dataset_copy = copy.deepcopy(self.dataset)
            log_odds[key] = self.get_log_odds_of_dataset(
                dataset=dataset_copy,
                attribute=attribute,
                apply_log=apply_log,
                attribution_baseline=attribution_baseline,
                masking_baseline=masking_baseline,
                num_samples=num_samples
            )[1]
        
        return log_odds
    
    def visualize_logs_odds_with_different_masking_baselines(
            self,
            attribute: Callable,
            attribution_baseline_str: str,
            apply_log: bool = False,
            num_samples: int = 100,
            title="",
            save_fig: bool = False
    ):
        log_odds_dict = self.get_log_odds_of_attribution_with_all_masking_baselines(attribute, attribution_baseline_str, apply_log, num_samples)
        visualize_logs_odds_with_different_masking_baselines(log_odds_dict, attribution_baseline_str, title, apply_log, save_fig=save_fig)
