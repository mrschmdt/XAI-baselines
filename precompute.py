import torch
import os
from evaluation.attribution_methods_evaluator_heloc import AttributionMethodsEvaluator
import pickle

BASE_DIR = os.getenv('BASE_DIR')

def precompute_feature_agreement_matrices_heloc(
    ks = list[int],
    rank_agreement: bool = False
)->None:
    for k in ks:
        model = torch.load(os.path.join(BASE_DIR, 'network', 'heloc_model.pth'))
        evaluator = AttributionMethodsEvaluator(model, dataset='HELOC')

        agreement_matrix = evaluator.get_feature_agreement_matrix(k, rank_agreement)
        pickle.dump(agreement_matrix, open(os.path.join(BASE_DIR, 'evaluation', 'precomputed', f'feature_agreement_matrix_k_{k}_rank_agreement_{rank_agreement}.pkl'), 'wb'))

def get_precomputed_feature_agreement_matrices_heloc(
    ks = list[int],
    rank_agreement: bool = False
)->dict:
    matrices = {}
    for k in ks:
        matrices[k] = pickle.load(open(os.path.join(BASE_DIR, 'evaluation', 'precomputed', f'feature_agreement_matrix_k_{k}_rank_agreement_{rank_agreement}.pkl'), 'rb'))
    return matrices