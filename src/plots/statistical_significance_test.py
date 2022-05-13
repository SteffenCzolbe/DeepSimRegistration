import pickle
import scipy.stats
from numpy import std, mean, sqrt
from .run_models import run_models


#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

def rank_metrics(results_dataset_dict):
    loss_functions = results[dataset].keys()
    mean_dice_overlaps = []
    for metric in loss_functions:
        mean_do = mean(results_dataset_dict[metric]["dice_overlap"])
        mean_dice_overlaps.append((metric, mean_do))
    mean_dice_overlaps.sort(reverse=True, key=lambda t: t[1])
    for rank, (metric,_) in enumerate(mean_dice_overlaps):
        results_dataset_dict[metric]["rank"] = rank
    return results_dataset_dict
    

def compare_metrics(results_dataset_dict, metric0, metric1):
    print(f'comparing {metric0} to {metric1}')
    scores_metric0 = results_dataset_dict[metric0]["dice_overlap"]
    scores_metric1 = results_dataset_dict[metric1]["dice_overlap"]
    
    _, p_value = scipy.stats.wilcoxon(
        scores_metric0,
        scores_metric1,
        alternative="greater")
        
    # perform Cohens' d 
    d = cohen_d(scores_metric0, scores_metric1)
    effect_size = "None"
    if abs(d) > 0.01:
        effect_size = "Very small"
    if abs(d) > 0.2:
        effect_size = "Small"
    if abs(d) > 0.5:
        effect_size = "Medium"
    if abs(d) > 0.8:
        effect_size = "Large"
    if abs(d) > 1.2:
        effect_size = "Vary large"
    if abs(d) > 2:
        effect_size = "Huge"
    
    if "compared_to" not in results_dataset_dict[metric0]:
        results_dataset_dict[metric0]["compared_to"] = {}
    results_dataset_dict[metric0]["compared_to"][metric1] = {} 
    results_dataset_dict[metric0]["compared_to"][metric1]["is_better"] = d > 0
    results_dataset_dict[metric0]["compared_to"][metric1]["p_val"] = p_value
    results_dataset_dict[metric0]["compared_to"][metric1]["cohens_d"] = d
    results_dataset_dict[metric0]["compared_to"][metric1]["cohens_d_category"] = effect_size
    return results_dataset_dict
    
def print_results(results):
    for dataset in results.keys():
        for loss_function in results[dataset].keys():
            rank = results[dataset][loss_function]["rank"]
            print(f"\n{dataset}, {loss_function} is rank {rank}:")
            for comparison_function in results[dataset][loss_function]["compared_to"].keys():
                relative_str = "is better" if results[dataset][loss_function]["compared_to"][comparison_function]["is_better"] else "is worse "
                p_value = results[dataset][loss_function]["compared_to"][comparison_function]["p_val"]
                d = results[dataset][loss_function]["compared_to"][comparison_function]["cohens_d"]
                effect_size = results[dataset][loss_function]["compared_to"][comparison_function]["cohens_d_category"]
                print(f'    {relative_str} than {comparison_function:15} p={p_value:.3f}, d={d:.2f} ({effect_size})')
    

if __name__ == '__main__':
    results = run_models(use_cached=True)

    for dataset in results.keys():
        results[dataset] = rank_metrics(results[dataset])
        loss_functions = results[dataset].keys()
        for metric0 in loss_functions:
            for metric1 in loss_functions:
                if metric0 == metric1:
                    continue
                # compare 2 metrics
                results[dataset] = compare_metrics(results[dataset], metric0, metric1)
        
    print_results(results)