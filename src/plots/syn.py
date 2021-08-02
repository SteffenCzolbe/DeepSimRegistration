import os
import yaml
import numpy as np
from . import config


def read_scores_from_yaml_file(dir):
    """reads the .yaml files in the dir
    """
    fnames = os.listdir(dir)
    scores = {}

    # read scores
    for fname in fnames:
        with open(os.path.join(dir, fname), 'r') as file:
            data = yaml.safe_load(file)
            feature_extractor = data['hparams']['feature_extractor']
            scores[feature_extractor] = data['scores']

    # filter out images where one model is missing a score (helps with partial evaluation during development)
    feature_extractors = scores.keys()
    idxs = [set(scores[fe].keys()) for fe in feature_extractors]
    common_idxs = set.intersection(* idxs)
    #print(f'found {len(common_idxs)} common samples for {dir}')
    for feature_extractor in feature_extractors:
        scores[feature_extractor] = np.array([scores[feature_extractor][idx]
                                              for idx in common_idxs])
    return scores


if __name__ == '__main__':

    for dataset in config.DATASET_ORDER:
        print(dataset)
        dataset_path = os.path.join('./out', dataset, 'syn')
        scores = read_scores_from_yaml_file(dataset_path)
        for feature_extractor in scores.keys():
            print(feature_extractor, np.median(scores[feature_extractor]))
