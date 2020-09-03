import pickle
import scipy.stats

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

pickle.dump(results, open(cache_file_name, "wb"))

