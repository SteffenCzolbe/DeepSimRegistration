from argparse import ArgumentParser
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from .config import *
from .run_models import run_models

def main(args):
    test_results = run_models(use_cached=True)
    data = defaultdict(list)

    for dataset in DATASET_ORDER:
        if dataset not in test_results.keys():
            continue
        for loss_function in LOSS_FUNTION_ORDER:
            if loss_function not in test_results[dataset].keys():
                continue          
            #print(dataset, loss_function)
            data['dataset'].append(dataset)
            data['loss_function'].append(loss_function)
            for k in test_results[dataset][loss_function]:
                v = test_results[dataset][loss_function][k]
                if isinstance(v, np.ndarray):
                    v_mean = v[~np.isnan(v)].mean(axis=0)
                    data[k].append(v_mean)

    df = pd.DataFrame(data)
    df.to_csv(args.output_file, sep=',', index=False, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--output_file', type=str, default='out/plots/metrics_all_baselines.csv', help='File with test results.')
    args = parser.parse_args()
    main(args)