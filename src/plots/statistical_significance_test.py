import pickle
import scipy.stats
from numpy import std, mean, sqrt

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

def compare_metrics(results_dataset_dict, metric0, metric1):
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
    
def print_results(results):
    for dataset in results.keys():
        for loss_function in results[dataset].keys():
            print(f"\n{dataset}, {loss_function}:")
            for comparison_function in results[dataset][loss_function]["compared_to"].keys():
                relative_str = "is better" if results[dataset][loss_function]["compared_to"][comparison_function]["is_better"] else "is worse "
                p_value = results[dataset][loss_function]["compared_to"][comparison_function]["p_val"]
                d = results[dataset][loss_function]["compared_to"][comparison_function]["cohens_d"]
                effect_size = results[dataset][loss_function]["compared_to"][comparison_function]["cohens_d_category"]
                print(f'    {relative_str} than {comparison_function:15} p={p_value:.3f}, d={d:.3f} ({effect_size})')
    

cache_file_name = "./src/plots/cache.pickl"


results = pickle.load(open(cache_file_name, "rb"))

for dataset in results.keys():
    loss_functions = results[dataset].keys()
    
    for metric0 in loss_functions:
        for metric1 in loss_functions:
            if metric0 == metric1:
                continue
            # compare
            compare_metrics(results[dataset], metric0, metric1)
    
print_results(results)
pickle.dump(results, open(cache_file_name, "wb"))

