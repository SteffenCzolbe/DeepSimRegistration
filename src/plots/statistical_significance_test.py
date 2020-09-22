import pickle
import scipy.stats
from numpy import std, mean, sqrt

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


cache_file_name = "./src/plots/cache.pickl"


results = pickle.load(open(cache_file_name, "rb"))

for dataset in results.keys():
    for loss_function in results[dataset].keys():
        if loss_function == "deepsim":
            results[dataset][loss_function][
                "statistically_significantly_worse_than_deepsim"
            ] = False
            results[dataset][loss_function][
                "statistically_significantly_worse_than_deepsim_pval"
            ] = 1
            results[dataset][loss_function][
                "cohens_d"
            ] = 0
            results[dataset][loss_function][
                "cohens_d_category"
            ] = "None"
            continue

        # perform one-sided Wilcoxon signed rank test for paired sample.
        # H_0: Models trained with DeepSim performs significantly better than model trained with other loss function
        _, p_value = scipy.stats.wilcoxon(
            results[dataset]["deepsim"]["dice_overlap"],
            results[dataset][loss_function]["dice_overlap"],
            alternative="greater",
        )

        # accept when statistically significant p < 0.05
        results[dataset][loss_function][
            "statistically_significantly_worse_than_deepsim_pval"
        ] = p_value
        results[dataset][loss_function][
            "statistically_significantly_worse_than_deepsim"
        ] = (p_value < 0.05)
        
        # perform Cohens' d 
        d = cohen_d(results[dataset]["deepsim"]["dice_overlap"], results[dataset][loss_function]["dice_overlap"])
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
            effect_size = "Hige"
        results[dataset][loss_function][
            "cohens_d"
        ] = d
        results[dataset][loss_function][
            "cohens_d_category"
        ] = effect_size
        print(f"Cohens' D for {dataset}, {loss_function}: {effect_size}, ({d})")
    

pickle.dump(results, open(cache_file_name, "wb"))

