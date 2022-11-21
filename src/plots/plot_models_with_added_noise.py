from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
from tqdm import tqdm
import pickle

from .config import *


if __name__ == '__main__':

    # set up sup-plots
    fig = plt.figure(figsize=(3, 3))
    ax = fig.subplots(1, 1)
    plt.subplots_adjust(bottom=0.33, wspace=0.275)
    
    dataset = 'platelet-em'
    
    # load data to plot
    cache_file_name = "./src/plots/cache_with_added_noise.pickl"
    with open(cache_file_name, 'rb') as f:
        results = pickle.load(f)
        
    # figure out x-ticks
    ticks = list(results[dataset]['l2'].keys())
    xs = np.arange(len(ticks))

    for loss_function in tqdm(
        LOSS_FUNTION_ORDER, desc=f"plotting convergence of loss-functions on {dataset}"
    ):
        if loss_function not in results[dataset]:
            continue
        
        ys = [results[dataset][loss_function][tick]['dice_overlap'].mean() for tick in ticks]
        
        c = LOSS_FUNTION_CONFIG[loss_function]['primary_color']
        line = ax.plot(
            xs, ys, color=c, linewidth=2
        )  
        # some trickery required to show lines on the legend of subplots not containing them
        LOSS_FUNTION_CONFIG[loss_function]["handle"] = line[0]

    # add labels
    ax.set_xlabel('Noise')
    ax.set_ylabel('Test Dice Overlap')
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in LOSS_FUNTION_ORDER
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in LOSS_FUNTION_ORDER
    ]

    legend = fig.legend([], [], loc="lower center",
            ncol=len(handles), handlelength=1, columnspacing=1)
    legend.get_frame().set_edgecolor("white")

    # configure precision
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax.set_xticks([0, 10000, 20000, 30000])

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)

    plt.savefig(f"./out/plots/pdf/added_noise.pdf", bbox_inches='tight')
    plt.savefig(f"./out/plots/png/added_noise.png", bbox_inches='tight')
