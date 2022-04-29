from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np

#######################
from .config import *
#from .config2D import *
#######################

from .run_models import run_models


def main(args):
    #load data

    # for config (3D data)
    # with open(args.cache_file_name, 'rb') as f:
    #     test_results = pickle.load(f)

    # for config2D (mind loss)
    test_results = run_models(use_cached=True)
    data = defaultdict(list)

    for dataset in DATASET_ORDER:
        if dataset not in test_results.keys():
            continue
        for loss_function in LOSS_FUNTION_ORDER:
            if loss_function not in test_results[dataset].keys():
                continue
            
            print(dataset, loss_function)

            data['dataset'].append(dataset)
            data['loss_function'].append(loss_function)
            for k in test_results[dataset][loss_function]:
                v = test_results[dataset][loss_function][k]
                if isinstance(v, np.ndarray):
                    #v = v.mean(axis=0)
                    v= v[~np.isnan(v)].mean(axis=0)
                data[k].append(v)

    # print
    #print(data.values())
    #print(test_results['brain-mri']['mind'].keys())
    #print(test_results['brain-mri']['mind']['jacobian_determinant_negative'])
    #print(test_results['brain-mri']['mind']['jacobian_determinant_log_var'])
    #print()
    # print(len(test_results['brain-mri']['mind']['jacobian_determinant_log_var']))
    # print(len(test_results['brain-mri']['l2']['jacobian_determinant_log_var']))

    mind_log_var = test_results['brain-mri']['mind']['jacobian_determinant_log_var']
    mind_det = test_results['brain-mri']['mind']['jacobian_determinant_negative']
    print()
    print(mind_log_var[~np.isnan(mind_log_var)].mean(axis=0))
    print(np.mean(test_results['brain-mri']['l2']['jacobian_determinant_log_var']))
    print()
    print(np.mean(test_results['brain-mri']['mind']['jacobian_determinant_negative']) * 100 )
    print(np.mean(test_results['brain-mri']['l2']['jacobian_determinant_negative']) * 100)


    df = pd.DataFrame(data)
    df.to_csv(args.output_file, sep=',', index=False, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument(
    #     '--cache_file_name', type=str, default='./src/plots/cache.pickl', help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, default='out/plots/metrics_mind_all.csv', help='File with test results.')
    args = parser.parse_args()
    main(args)
