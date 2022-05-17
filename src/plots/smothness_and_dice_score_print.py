""" create a scatterplot of dice overlap vs transformation smoothness.
    """
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np
from .config2D import *
import os
from matplotlib.ticker import FormatStrFormatter

def load_data_for_model(dataset, loss_function):
    # load data
    with open(args.cache_file_name, 'rb') as f:
        test_results = pickle.load(f)
    if dataset not in test_results.keys():
        return None, None
    if loss_function not in test_results[dataset].keys():
        return None, None
    dice = test_results[dataset][loss_function]["dice_overlap"].mean(axis=0)
    log_var = test_results[dataset][loss_function]["jacobian_determinant_log_var"]
    smoothness = log_var[~np.isnan(log_var)].mean(axis=0)
    folding = test_results[dataset][loss_function]["jacobian_determinant_negative"].mean(axis=0)
    return dice, smoothness, folding


def main(args):
    for i, dataset in enumerate(DATASET_ORDER):
        for loss_function in EXTRACT_ZERO_MEAN_LOSS_FUNCTIONS:
            dice, smoothness , folding = load_data_for_model(dataset, loss_function)
            print(dataset, loss_function, np.round(dice,3), np.round(smoothness,2), np.round(folding * 100,2))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--cache_file_name', type=str, default='./src/plots/cache2D_zero_mean.pickl', help='File with test results.')
    args = parser.parse_args()
    main(args)
