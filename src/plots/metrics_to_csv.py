from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np
from .config import *


def main(args):
    # load data
    with open(args.cache_file_name, 'rb') as f:
        test_results = pickle.load(f)

    data = defaultdict(list)

    for dataset in DATASET_ORDER:
        if dataset not in test_results.keys():
            continue
        for loss_function in LOSS_FUNTION_ORDER:
            if loss_function not in test_results[dataset].keys():
                continue

            data['dataset'].append(dataset)
            data['loss_function'].append(loss_function)
            for k in test_results[dataset][loss_function]:
                v = test_results[dataset][loss_function][k]
                if isinstance(v, np.ndarray):
                    v = v.mean(axis=0)
                data[k].append(v)

    # print
    df = pd.DataFrame(data)
    df.to_csv(args.output_file, sep=',', index=False, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--cache_file_name', type=str, default='./src/plots/cache.pickl', help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, default='out/plots/metrics.csv', help='File with test results.')
    args = parser.parse_args()
    main(args)
